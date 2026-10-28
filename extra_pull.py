import argparse
import csv
import json
import os
from calendar import monthrange
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import collect


OVERSHOOT = Decimal("1.10")
PRICE = Decimal(str(collect.PRICE_PER_1000)) / 1000
ITEMS_PER_DAY = 40
NOTES = [
    "Each day-window returns the newest items of that UTC day, so the last hours of the UTC day (Indonesian night) are over-represented; sub-day bounds do not work.",
    "New posts must carry a pull_batch marker (day_sample) when merged; the earlier month-wide pull is popularity-skewed.",
]


def keyword_order(rows):
    return list(dict.fromkeys(row["keyword"] for row in rows))


def sample_days(days_in_month, days_per_cell):
    if days_in_month < 1 or days_per_cell < 1:
        raise ValueError("Month length and days per cell must be positive")
    count = min(days_per_cell, days_in_month)
    days = [1 + (2 * i + 1) * days_in_month // (2 * count) for i in range(count)]
    assert len(days) == len(set(days)) == count
    assert all(1 <= day <= days_in_month for day in days)
    gaps = [b - a for a, b in zip(days, days[1:])]
    assert not gaps or max(gaps) - min(gaps) <= 1
    return days


def quote_term(term):
    return f'"{term}"' if " " in term else term


def build_query(keyword, earlier, day):
    terms = quote_term(keyword) + "".join(" -" + quote_term(term) for term in earlier)
    next_day = day + timedelta(days=1)
    return f"{terms} since:{day.isoformat()}_00:00:00_UTC until:{next_day.isoformat()}_00:00:00_UTC"


def plan_cells(rows, days_per_cell):
    keywords = keyword_order(rows)
    cells = []
    for position, keyword in enumerate(keywords):
        capped = sorted((row for row in rows
                         if row["keyword"] == keyword and row["capped"] == "True"),
                        key=lambda row: row["month"])
        for row in capped:
            start = date.fromisoformat(f"{row['month']}-01")
            days = sample_days(monthrange(start.year, start.month)[1], days_per_cell)
            cells.append({
                "keyword": keyword, "month": row["month"],
                "sampled_days": days,
                "queries": [build_query(keyword, keywords[:position], start.replace(day=day))
                            for day in days],
            })
    queries = [query for cell in cells for query in cell["queries"]]
    assert len(queries) == len(set(queries)), "Planned queries must be unique"
    suffix = "_00:00:00_UTC"
    for query in queries:
        since, until = query.rsplit(" since:", 1)[1].split(" until:")
        assert since.endswith(suffix) and until.endswith(suffix)
        assert date.fromisoformat(until.removesuffix(suffix)) == (
            date.fromisoformat(since.removesuffix(suffix)) + timedelta(days=1))
    return cells


def make_chunks(cells, items_per_day=ITEMS_PER_DAY):
    if items_per_day < 20:
        raise ValueError("--items-per-day must be at least 20 (the actor minimum)")
    chunks = []
    for keyword in keyword_order(cells):
        ordered = sorted((cell for cell in cells if cell["keyword"] == keyword),
                         key=lambda cell: cell["month"])
        for offset in range(0, len(ordered), 6):
            group = ordered[offset:offset + 6]
            queries = [query for cell in group for query in cell["queries"]]
            items = len(queries) * items_per_day
            actor_input = collect.build_actor_input(queries, items)
            assert actor_input["maxItems"] == items_per_day
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


def choose_plan(rows, items_per_day, remaining, reserve, requested="auto"):
    if requested == "auto":
        candidates = range(14, 0, -1)
    else:
        days = int(requested)
        if days < 1:
            raise ValueError("--days-per-cell must be auto or a positive integer")
        candidates = [days]
    for days in candidates:
        cells = plan_cells(rows, days)
        chunks = make_chunks(cells, items_per_day)
        committed = assign_chunks(chunks, remaining, reserve)
        if requested != "auto" or days == 1 or all(
                chunk["token_index"] is not None for chunk in chunks):
            return days, cells, chunks, committed


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


def write_plan(cells, chunks, days_per_cell, items_per_day, remaining, reserve):
    fields = ["keyword", "month", "sampled_days", "queries", "items_per_day",
              "items_cap", "cost_bound_usd"]
    with Path("data/extra_pull_plan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            items = len(cell["queries"]) * items_per_day
            writer.writerow({
                "keyword": cell["keyword"], "month": cell["month"],
                "sampled_days": ";".join(map(str, cell["sampled_days"])),
                "queries": len(cell["queries"]), "items_per_day": items_per_day,
                "items_cap": items, "cost_bound_usd": f"{items * PRICE:.6f}",
            })
    parameters = {"days_per_cell": days_per_cell, "items_per_day": items_per_day,
                  "reserve": reserve, "price": PRICE,
                  "overshoot": OVERSHOOT, "tokens": list(remaining), "remaining": remaining}
    Path("data/extra_pull_plan.json").write_text(
        json.dumps({"parameters": parameters, "notes": NOTES, "chunks": chunks},
                   ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")


def print_plan(keywords, cells, chunks, committed, remaining, reserve, days_per_cell, items_per_day):
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
        print(f"{keyword:<{width}} {count:7d} {queries:7d} {items_per_day:11d} "
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
    print(f"Chosen D: {days_per_cell} ({len(cells)} capped cells, {len(chunks)} chunks)")
    print("Dry run: nothing fetched, nothing spent.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plan extra MBG pulls without fetching or spending.")
    parser.add_argument("--days-per-cell", default="auto", help="sampled days per capped cell, or auto")
    parser.add_argument("--items-per-day", type=int, default=ITEMS_PER_DAY, help="items per day query (minimum 20)")
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
        if args.items_per_day < 20:
            raise ValueError("--items-per-day must be at least 20 (the actor minimum)")
        if args.days_per_cell != "auto" and int(args.days_per_cell) < 1:
            raise ValueError("--days-per-cell must be auto or a positive integer")
        with Path("data/collection_log.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        remaining = read_remaining(indices, args.remaining)
        days, cells, chunks, committed = choose_plan(
            rows, args.items_per_day, remaining, reserve, args.days_per_cell)
    except (ValueError, InvalidOperation) as error:
        parser.error(str(error))
    write_plan(cells, chunks, days, args.items_per_day, remaining, reserve)
    print_plan(keyword_order(rows), cells, chunks, committed, remaining, reserve, days, args.items_per_day)


if __name__ == "__main__":
    main()
