import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import collect


LOG_FIELDS = [
    "chunk", "keyword", "months", "token_index", "queries", "queries_hash",
    "items_returned", "kept_new", "dropped_lang_or_rt", "dropped_duplicate",
    "dropped_out_of_window", "excluded_term_hits", "malformed", "est_cost_usd",
    "started_at", "finished_at", "status", "error",
]
COUNT_FIELDS = [
    "items_returned", "kept_new", "dropped_lang_or_rt", "dropped_duplicate",
    "dropped_out_of_window", "excluded_term_hits", "malformed",
]


def parse_chunks(raw, total):
    if raw is None:
        return list(range(1, total + 1))
    selected = set()
    for part in raw.split(","):
        match = re.fullmatch(r"\s*([0-9]+)(?:\s*-\s*([0-9]+))?\s*", part)
        if not match:
            raise ValueError("--chunks must be a comma list of chunk numbers or ranges")
        first = int(match[1])
        last = int(match[2]) if match[2] else first
        if not 1 <= first <= last <= total:
            raise ValueError(f"--chunks must be within 1-{total}, with ascending ranges")
        selected.update(range(first, last + 1))
    return sorted(selected)


def queries_hash(queries):
    return hashlib.sha256("\n".join(queries).encode("utf-8")).hexdigest()[:16]


def read_credit(token):
    from apify_client import ApifyClient

    usage = ApifyClient(token).user().limits()
    return (Decimal(str(usage.limits.max_monthly_usage_usd))
            - Decimal(str(usage.current.monthly_usage_usd)))


def amount(value):
    value = Decimal(str(value))
    if not value.is_finite():
        raise ValueError("Credit and bounds must be finite")
    return value


def redact(message, tokens):
    for token in sorted(tokens, key=len, reverse=True):
        message = message.replace(token, "<token>")
    return message


def safe_error(error, tokens):
    # Redact before truncating so a token at the cutoff cannot leak a prefix.
    message = redact(str(error), tokens).replace("\r", " ").replace("\n", " ")
    return f"{type(error).__name__}: {message[:120]}"


def query_info(query):
    bounds = re.search(r" since_time:([0-9]+) until_time:([0-9]+)$", query)
    if not bounds:
        raise ValueError("Queries must end with integer since_time/until_time bounds")
    since, until = map(int, bounds.groups())
    if since >= until:
        raise ValueError("Query windows must have since_time < until_time")
    start, end = (datetime.fromtimestamp(value, timezone.utc) for value in (since, until))
    terms = [quoted or bare for quoted, bare in
             re.findall(r'(?:^|\s)-(?:"([^"]+)"|(\S+))', query)]
    excluded = (re.compile(r"(?<!\w)(?:" + "|".join(re.escape(term) for term in terms)
                           + r")(?!\w)", re.IGNORECASE) if terms else None)
    return start, end, excluded


def validate_chunk(chunk, token_count):
    if not isinstance(chunk["keyword"], str) or not chunk["keyword"]:
        raise ValueError("Chunk keyword must be nonempty")
    if (not isinstance(chunk["months"], list) or not chunk["months"]
            or any(not isinstance(month, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}", month)
                   for month in chunk["months"])):
        raise ValueError("Chunk months must be a nonempty YYYY-MM list")
    index = chunk["token_index"]
    if type(index) is not int or not 1 <= index <= token_count:
        raise ValueError("Chunk token_index must refer to an available APIFY_TOKEN position")
    queries = chunk["queries"]
    if not isinstance(queries, list) or not queries or any(not isinstance(q, str) for q in queries):
        raise ValueError("Chunk queries must be a nonempty string list")
    cap = chunk["items_cap"]
    if type(cap) is not int or cap < len(queries) * 20:
        raise ValueError("Chunk items_cap must cover the actor minimum of 20 items per query")
    cost_bound, safe_bound = amount(chunk["cost_bound"]), amount(chunk["safe_bound"])
    expected = Decimal(cap) * Decimal(str(collect.PRICE_PER_1000)) / 1000
    if not safe_bound >= cost_bound >= expected:
        raise ValueError("Chunk bounds must cover its items_cap at the collection price")
    return {query: query_info(query) for query in queries}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def keep_items(items, salt, queries, info, keyword, seen, log):
    rows = []
    for item in items:
        try:
            row = collect.minimize(item, salt, queries)
            if row is None:
                log["dropped_lang_or_rt"] += 1
                continue
            if not isinstance(row["id"], str) or not row["id"]:
                raise ValueError("Invalid post id")
            start, end, excluded = info[row["query"]]
            created = datetime.fromisoformat(row["created_at"])
        except (AttributeError, KeyError, TypeError, ValueError):
            log["malformed"] += 1
            continue
        if row["id"] in seen:
            log["dropped_duplicate"] += 1
            continue
        if not start <= created < end:
            log["dropped_out_of_window"] += 1
            continue
        if excluded and excluded.search(row["text"]):
            log["excluded_term_hits"] += 1
        row.update(pull_batch="slot_sample", source_keyword=keyword,
                   source_month=start.strftime("%Y-%m"), slot_since=start.isoformat(),
                   slot_until=end.isoformat())
        rows.append(row)
        seen.add(row["id"])
    log["kept_new"] = len(rows)
    return rows


def main(argv=None, *, fetch_fn=None, credit_fn=None, data_dir="data"):
    parser = argparse.ArgumentParser(description="Execute planned extra pulls using real Apify credit.")
    parser.add_argument("--yes", action="store_true", help="authorize spending on selected chunks")
    parser.add_argument("--plan", type=Path, default=Path("data/extra_pull_plan.json"))
    parser.add_argument("--chunks", help="1-based chunk numbers/ranges, e.g. 19,1-3 (plan order)")
    args = parser.parse_args(argv)
    if not args.yes:
        print("Refusing to spend: --yes is required. Nothing fetched, nothing spent.", flush=True)
        return 1
    missing = [name for name in ("APIFY_TOKEN", "USER_HASH_SALT") if not os.environ.get(name)]
    if missing:
        parser.error("Missing required environment variables: " + ", ".join(missing))
    try:
        tokens = collect.parse_tokens(os.environ["APIFY_TOKEN"])
    except ValueError as error:
        parser.error(str(error))
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        chunks = plan["chunks"]
        if not isinstance(chunks, list):
            raise ValueError("Plan chunks must be a list")
        selected = parse_chunks(args.chunks, len(chunks))
        info = {number: validate_chunk(chunks[number - 1], len(tokens)) for number in selected}
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        log_path = data_dir / "extra_pull_log.csv"
        if log_path.exists() and log_path.stat().st_size:
            with log_path.open(newline="", encoding="utf-8") as stream:
                if next(csv.reader(stream)) != LOG_FIELDS:
                    raise ValueError("Existing extra_pull_log.csv has unexpected columns")
        history = collect._read_csv(log_path)
        completed = {(row["chunk"], row["queries_hash"]) for row in history if row["status"] == "ok"}
        # collect's reader splits on literal newlines, preserving U+2028 in tweet text.
        seen = {row["id"] for name in ("final_posts.jsonl", "extra_posts.jsonl")
                for row in collect._read_jsonl(data_dir / name)}
    except (OSError, KeyError, TypeError, ValueError, InvalidOperation, OverflowError) as error:
        parser.error(redact(f"Cannot execute plan: {type(error).__name__}: {error}", tokens))

    totals = {}
    for number in selected:
        chunk = chunks[number - 1]
        group = totals.setdefault(chunk["token_index"], [0, Decimal(0), Decimal(0)])
        group[0] += chunk["items_cap"]
        group[1] += amount(chunk["cost_bound"])
        group[2] += amount(chunk["safe_bound"])
    print(f"Selected chunks: {len(selected)}; items cap {sum(t[0] for t in totals.values())}; "
          f"total cost bound ${sum(t[1] for t in totals.values()):.6f}; "
          f"total safe bound ${sum(t[2] for t in totals.values()):.6f}", flush=True)
    for index, (cap, bound, safe) in sorted(totals.items()):
        print(f"token #{index}: items cap {cap}; cost bound ${bound:.6f}; safe bound ${safe:.6f}",
              flush=True)

    fetch_fn = collect.fetch if fetch_fn is None else fetch_fn
    credit_fn = read_credit if credit_fn is None else credit_fn
    statuses, used = Counter(), set()
    logs, already_ok = [], 0
    for number in selected:
        chunk = chunks[number - 1]
        queries, index = chunk["queries"], chunk["token_index"]
        digest = queries_hash(queries)
        label = (f"chunk {number}/{len(chunks)} {chunk['keyword']} "
                 f"{';'.join(chunk['months'])} token #{index}")
        if (str(number), digest) in completed:
            already_ok += 1
            print(f"{label}: already ok (same queries_hash)", flush=True)
            continue
        log = dict.fromkeys(COUNT_FIELDS, 0)
        log.update(chunk=number, keyword=chunk["keyword"], months=";".join(chunk["months"]),
                   token_index=index, queries=len(queries), queries_hash=digest,
                   est_cost_usd=0.0, started_at=utc_now(), status="ok", error="")
        token = tokens[index - 1]
        used.add(index)
        try:
            remaining = amount(credit_fn(token))
        except Exception as error:
            log.update(status="skipped", error="credit lookup " + safe_error(error, tokens))
        else:
            if remaining < amount(chunk["safe_bound"]) + Decimal("0.02"):
                log.update(status="skipped", error="credit")
        if log["status"] == "ok":
            try:
                items = fetch_fn(queries, chunk["items_cap"], token)
            except Exception as error:
                log.update(status="failed", error=safe_error(error, tokens))
            else:
                log["items_returned"] = len(items)
                log["est_cost_usd"] = collect.cost(len(items))
                rows = keep_items(items, os.environ["USER_HASH_SALT"], queries, info[number],
                                  chunk["keyword"], seen, log)
                collect._append_jsonl(data_dir / "extra_posts.jsonl", rows)
        log["finished_at"] = utc_now()
        collect._append_csv(log_path, LOG_FIELDS, [log])
        logs.append(log)
        statuses[log["status"]] += 1
        print(f"{label}: items {log['items_returned']} kept {log['kept_new']} "
              f"dup {log['dropped_duplicate']} lang/rt {log['dropped_lang_or_rt']} "
              f"out_of_window {log['dropped_out_of_window']} excluded {log['excluded_term_hits']} "
              f"malformed {log['malformed']} est ${log['est_cost_usd']:.4f} {log['status']}"
              + (f" ({log['error']})" if log["error"] else ""), flush=True)

    print(f"Summary: ok {statuses['ok']} failed {statuses['failed']} skipped {statuses['skipped']} "
          f"already_ok {already_ok}; total items returned {sum(r['items_returned'] for r in logs)}; "
          f"total kept {sum(r['kept_new'] for r in logs)}; "
          f"total estimated cost ${sum(r['est_cost_usd'] for r in logs):.6f}", flush=True)
    for status in ("failed", "skipped"):
        numbers = ",".join(str(row["chunk"]) for row in logs if row["status"] == status)
        print(f"{status.capitalize()} chunks: {numbers or 'none'}", flush=True)
    for index in sorted(used):
        try:
            remaining = f"${amount(credit_fn(tokens[index - 1])):.6f}"
        except Exception:
            remaining = "unknown"
        print(f"Remaining credit token #{index}: {remaining}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
