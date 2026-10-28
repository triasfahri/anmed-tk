import argparse
import csv
import json
import os
import re
from calendar import monthrange, timegm
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import collect


OVERSHOOT = Decimal("1.10")
PRICE = Decimal(str(collect.PRICE_PER_1000)) / 1000
ITEMS_PER_SLOT = 20
SLOT_HOURS = 6
ROTATION_STEP = 5
PHASE_STEP = 7
NOTES = [
    "Each slot returns the newest items inside its 6-hour UTC window; busy slots return a cluster of consecutive posts. Slot start hours rotate so every hour of the day is covered.",
    "The actor cannot filter language: non-Indonesian items are paid for and dropped.",
    "New posts must carry a pull_batch marker (slot_sample) when merged; the earlier month-wide pull is popularity-skewed.",
]


def keyword_order(rows):
    return list(dict.fromkeys(row["keyword"] for row in rows))


def month_bounds(month):
    start = datetime.fromisoformat(f"{month}-01").replace(tzinfo=timezone.utc)
    return start, start + timedelta(days=monthrange(start.year, start.month)[1])


def sample_slots(month, slots_per_cell):
    start, following = month_bounds(month)
    length = (following - start).days
    if not 1 <= slots_per_cell <= length:
        raise ValueError(f"--slots-per-cell must be between 1 and {length} for {month}")
    # Integer arithmetic implements floor((k + 0.5) * L / D) exactly.
    days = [(2 * k + 1) * length // (2 * slots_per_cell) for k in range(slots_per_cell)]
    gaps = [b - a for a, b in zip(days, days[1:])]
    assert not gaps or max(gaps) - min(gaps) <= 1
    month_index = (start.year - 2025) * 12 + start.month - 1
    phase = month_index * PHASE_STEP % 24
    duration = timedelta(hours=SLOT_HOURS)
    slots = []
    for k, day in enumerate(days):
        hour = (phase + k * ROTATION_STEP) % 24
        slot_start = min(start + timedelta(days=day, hours=hour), following - duration)
        slots.append((slot_start, slot_start + duration))
    assert len({a for a, _ in slots}) == slots_per_cell, "Slot starts must be distinct"
    assert all(start <= a < b <= following and b - a == duration for a, b in slots)
    if slots_per_cell >= 8:
        assert len({a.hour for a, _ in slots}) >= 8, "Slot hours must rotate"
    return slots


def quote_term(term):
    return f'"{term}"' if " " in term else term


def build_query(keyword, earlier, start, end):
    terms = quote_term(keyword) + "".join(" -" + quote_term(term) for term in earlier)
    return f"{terms} since_time:{timegm(start.utctimetuple())} until_time:{timegm(end.utctimetuple())}"


def plan_cells(rows, slots_per_cell):
    keywords = keyword_order(rows)
    cells = []
    for position, keyword in enumerate(keywords):
        capped = sorted((row for row in rows
                         if row["keyword"] == keyword and row["capped"] == "True"),
                        key=lambda row: row["month"])
        for row in capped:
            slots = sample_slots(row["month"], slots_per_cell)
            queries = [build_query(keyword, keywords[:position], start, end)
                       for start, end in slots]
            month_start, month_end = month_bounds(row["month"])
            starts = []
            for query in queries:
                bounds = re.search(r" since_time:([0-9]+) until_time:([0-9]+)$", query)
                assert bounds, "Queries must use integer epoch bounds"
                since, until = map(int, bounds.groups())
                assert until - since == 21600, "Slots must last exactly six hours"
                assert timegm(month_start.utctimetuple()) <= since
                assert until <= timegm(month_end.utctimetuple())
                starts.append(since)
            assert len(starts) == len(set(starts)), "Slot starts must be distinct"
            cells.append({
                "keyword": keyword, "month": row["month"],
                "slot_starts_utc": [start.strftime("%Y-%m-%dT%H:%M") for start, _ in slots],
                "queries": queries,
            })
    queries = [query for cell in cells for query in cell["queries"]]
    assert len(queries) == len(set(queries)), "Planned queries must be unique"
    return cells


def make_chunks(cells, items_per_slot=ITEMS_PER_SLOT):
    if items_per_slot < 20:
        raise ValueError("--items-per-slot must be at least 20 (the actor minimum)")
    chunks = []
    for keyword in keyword_order(cells):
        ordered = sorted((cell for cell in cells if cell["keyword"] == keyword),
                         key=lambda cell: cell["month"])
        for offset in range(0, len(ordered), 6):
            group = ordered[offset:offset + 6]
            queries = [query for cell in group for query in cell["queries"]]
            items = len(queries) * items_per_slot
            actor_input = collect.build_actor_input(queries, items)
            assert actor_input["maxItems"] == items_per_slot
            chunks.append({
                "keyword": keyword, "months": [cell["month"] for cell in group],
                "queries": queries, "actor_input": actor_input, "token_index": None,
                "items_cap": items, "cost_bound": items * PRICE,
                "safe_bound": items * PRICE * OVERSHOOT,
            })
    assert all(len(chunk["months"]) <= 6 for chunk in chunks)
    return chunks


def assign_chunks(chunks, remaining, reserve):
    committed = dict.fromkeys(remaining, Decimal(0))
    for chunk in chunks:
        chunk["token_index"] = None
    for chunk in sorted(chunks, key=lambda chunk: chunk["safe_bound"], reverse=True):
        token = max(remaining, key=lambda index: remaining[index] - reserve - committed[index])
        if remaining[token] - reserve - committed[token] >= chunk["safe_bound"]:
            chunk["token_index"] = token
            committed[token] += chunk["safe_bound"]
    return committed


def choose_plan(rows, items_per_slot, remaining, reserve, requested="12"):
    max_slots = 31
    for row in rows:
        if row["capped"] == "True":
            start, end = month_bounds(row["month"])
            max_slots = min(max_slots, (end - start).days)
    if requested == "auto":
        candidates = range(max_slots, 0, -1)
    else:
        slots = int(requested)
        if not 1 <= slots <= max_slots:
            raise ValueError(f"--slots-per-cell must be auto or between 1 and {max_slots}")
        candidates = [slots]
    for slots in candidates:
        cells = plan_cells(rows, slots)
        chunks = make_chunks(cells, items_per_slot)
        committed = assign_chunks(chunks, remaining, reserve)
        if requested != "auto" or slots == 1 or all(
                chunk["token_index"] is not None for chunk in chunks):
            return slots, cells, chunks, committed


def read_remaining(indices, raw):
    if raw is not None:
        amounts = [Decimal(value.strip()) for value in raw.split(",")]
        if len(amounts) != len(indices):
            raise ValueError("--remaining must have one amount per selected token")
        if any(not amount.is_finite() or amount < 0 for amount in amounts):
            raise ValueError("--remaining amounts must be finite and nonnegative")
        return dict(zip(indices, amounts))

    from apify_client import ApifyClient

    tokens = collect.parse_tokens(os.environ.get("APIFY_TOKEN", ""))
    if max(indices) > len(tokens):
        raise ValueError(f"APIFY_TOKEN has no entry #{max(indices)}")
    remaining = {}
    for index in indices:
        try:
            usage = ApifyClient(tokens[index - 1]).user().limits()
            remaining[index] = (Decimal(str(usage.limits.max_monthly_usage_usd))
                                - Decimal(str(usage.current.monthly_usage_usd)))
            if not remaining[index].is_finite():
                raise ValueError
        except Exception:
            raise ValueError(f"Cannot read credit for token #{index}; use --remaining offline") from None
    return remaining


def write_plan(cells, chunks, slots_per_cell, items_per_slot, remaining, reserve):
    fields = ["keyword", "month", "slots", "slot_starts_utc", "items_per_slot",
              "items_cap", "cost_bound_usd"]
    with Path("data/extra_pull_plan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            items = len(cell["queries"]) * items_per_slot
            writer.writerow({
                "keyword": cell["keyword"], "month": cell["month"],
                "slots": len(cell["queries"]),
                "slot_starts_utc": ";".join(cell["slot_starts_utc"]),
                "items_per_slot": items_per_slot,
                "items_cap": items, "cost_bound_usd": f"{items * PRICE:.6f}",
            })
    parameters = {"slots_per_cell": slots_per_cell, "items_per_slot": items_per_slot,
                  "slot_hours": SLOT_HOURS, "rotation_step": ROTATION_STEP, "phase_step": PHASE_STEP,
                  "reserve": reserve, "price": PRICE,
                  "overshoot": OVERSHOOT, "tokens": list(remaining), "remaining": remaining}
    Path("data/extra_pull_plan.json").write_text(
        json.dumps({"parameters": parameters, "notes": NOTES, "chunks": chunks},
                   ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")


def print_plan(keywords, cells, chunks, committed, remaining, reserve, slots_per_cell, items_per_slot):
    width = max([len("Keyword")] + [len(keyword) for keyword in keywords])
    print(f"{'Keyword':<{width}} {'Cells':>7} {'Queries':>7} {'Items/query':>11} "
          f"{'Items cap':>9} {'Cost USD':>10} {'Safe USD':>10}")
    for keyword in keywords + ["TOTAL"]:
        group = [chunk for chunk in chunks if keyword == "TOTAL" or chunk["keyword"] == keyword]
        count = sum(len(chunk["months"]) for chunk in group)
        queries = sum(len(chunk["queries"]) for chunk in group)
        items = sum(chunk["items_cap"] for chunk in group)
        cost = sum((chunk["cost_bound"] for chunk in group), Decimal(0))
        safe = sum((chunk["safe_bound"] for chunk in group), Decimal(0))
        print(f"{keyword:<{width}} {count:7d} {queries:7d} {items_per_slot:11d} "
              f"{items:9d} {cost:10.6f} {safe:10.6f}")
    print()
    print(f"{'Token':>5} {'Committed safe USD':>18} {'Remaining USD':>13} {'Reserve USD':>11}")
    for index, amount in remaining.items():
        print(f"{index:5d} {committed[index]:18.6f} {amount:13.6f} {reserve:11.6f}")
    unassigned = [chunk for chunk in chunks if chunk["token_index"] is None]
    if unassigned:
        safe = sum(chunk["safe_bound"] for chunk in unassigned)
        free = max(amount - reserve - committed[index] for index, amount in remaining.items())
        gap = max(Decimal(0), max(chunk["safe_bound"] for chunk in unassigned) - free)
        print(f"Shortfall: {len(unassigned)} unassigned chunks (${safe:.6f} safe bound); "
              f"${gap:.6f} more free on one token would fit the largest.")
    print(f"Chosen slots per cell: {slots_per_cell}")
    print("Dry run: nothing fetched, nothing spent.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plan extra MBG pulls without fetching or spending.")
    parser.add_argument("--slots-per-cell", default="12", help="six-hour slots per capped cell, or auto")
    parser.add_argument("--items-per-slot", type=int, default=ITEMS_PER_SLOT, help="items per slot query (minimum 20)")
    parser.add_argument("--tokens", default="1,3,4", help="1-based APIFY_TOKEN positions")
    parser.add_argument("--remaining", help="remaining USD per selected token; skips the Apify API")
    parser.add_argument("--reserve", default="0.20", help="USD to keep unspent per token")
    args = parser.parse_args(argv)
    try:
        indices = [int(value.strip()) for value in args.tokens.split(",")]
        if not indices or min(indices) < 1 or len(indices) != len(set(indices)):
            raise ValueError("--tokens must contain distinct positive indices")
        reserve = Decimal(args.reserve)
        if not reserve.is_finite() or reserve < 0:
            raise ValueError("--reserve must be finite and nonnegative")
        if args.items_per_slot < 20:
            raise ValueError("--items-per-slot must be at least 20 (the actor minimum)")
        if args.slots_per_cell != "auto" and int(args.slots_per_cell) < 1:
            raise ValueError("--slots-per-cell must be auto or a positive integer")
        with Path("data/collection_log.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        remaining = read_remaining(indices, args.remaining)
        slots, cells, chunks, committed = choose_plan(
            rows, args.items_per_slot, remaining, reserve, args.slots_per_cell)
    except (ValueError, InvalidOperation) as error:
        parser.error(str(error))
    write_plan(cells, chunks, slots, args.items_per_slot, remaining, reserve)
    print_plan(keyword_order(rows), cells, chunks, committed, remaining, reserve, slots, args.items_per_slot)


if __name__ == "__main__":
    main()
