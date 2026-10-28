import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

ACTOR_ID = "kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest"  # id CJdippxWmn9uRfooo, the actor behind fixtures/sample_tweet.json
# User-chosen keywords; X search is case-insensitive.
KEYWORDS = ["mbg", "makan bergizi gratis", "sppg", "satuan pelayanan pemenuhan gizi", "bgn",
            "badan gizi nasional", "embege"]
START, END = "2025-01-01", "2026-10-26"
MAX_PER_QUERY_FINAL = 300  # per (term, month); 7 terms x 22 months x 300 = $11.55 worst case at $0.25/1k
PRICE_PER_1000 = 0.25  # worst case, the actor's FREE-plan tier; paid plans go down to 0.18


def minimize(item, salt, queries=None):
    if item.get("retweeted_tweet") or item.get("lang") not in {"in", "id"}:
        return None
    author = item.get("author") or {}
    quote = item.get("quoted_tweet") or item.get("quoted_tweet_results")
    kind = "quote" if quote else "reply" if item.get("isReply") else "original"
    if quote:
        quote = quote.get("result", quote)
        quote = quote.get("tweet", quote)
    index = item.get("searchTermIndex")
    query = None
    if queries is not None and isinstance(index, int) and 0 <= index < len(queries):
        query = queries[index]
    return {
        "id": item["id"],
        "text": re.sub(r"@\w+", "@pengguna", item["text"]),
        "created_at": datetime.strptime(
            item["createdAt"], "%a %b %d %H:%M:%S %z %Y"
        ).astimezone(timezone.utc).isoformat(),
        "likes": item.get("likeCount", 0),
        "retweets": item.get("retweetCount", 0),
        "replies": item.get("replyCount", 0),
        "quotes": item.get("quoteCount", 0),
        "type": kind,
        "parent_id": (quote.get("id") or quote.get("rest_id")) if quote else (
            item.get("inReplyToId") if kind == "reply" else None
        ),
        "lang": item["lang"],
        "user_hash": hashlib.sha256((salt + str(author["id"])).encode()).hexdigest()[:16],
        "bio": (author.get("profile_bio") or {}).get("description")
        or author.get("description") or "",
        "followers": author.get("followers", 0),
        "query": query,
    }


def month_windows(start, end):
    current, end = date.fromisoformat(start), date.fromisoformat(end)
    windows = []
    while current < end:
        following = date(current.year + current.month // 12, current.month % 12 + 1, 1)
        following = min(following, end)
        windows.append((current.isoformat(), following.isoformat()))
        current = following
    return windows


def build_queries(term, windows):
    term = f'"{term}"' if " " in term else term
    # native retweets are excluded by this actor unless include:nativeretweets is set.
    # Plain YYYY-MM-DD dates leak hours into the next day; the _00:00:00_UTC form is exact (checked live).
    return [f"{term} since:{a}_00:00:00_UTC until:{b}_00:00:00_UTC" for a, b in windows]


def build_actor_input(queries, max_items):
    # this actor applies maxItems PER search term (minimum 20), so split the total across queries
    per_term = max(20, -(-max_items // len(queries)))
    return {"searchTerms": queries, "maxItems": per_term, "queryType": "Latest", "lang": "in"}


def fetch(queries, max_items, token):
    from apify_client import ApifyClient

    client = ApifyClient(token)
    run = client.actor(ACTOR_ID).call(
        run_input=build_actor_input(queries, max_items), max_items=max_items, logger=None
    )
    status = run.status if run is not None else "None"
    status_message = run.status_message if run is not None else None
    # the platform cap (max_items x price) can cut a run just short of maxItems; those items are already paid for
    capped = status == "ABORTED" and "maximum cost" in (status_message or "")
    if status != "SUCCEEDED" and not capped:
        raise RuntimeError(f"Apify run {status}: {status_message}")
    return list(client.dataset(run.default_dataset_id).iterate_items())


def parse_tokens(raw) -> list[str]:
    tokens = []
    for position, value in enumerate(raw.split(","), 1):
        token = value.strip().strip("'\"").strip()
        if not token:
            continue
        if token.lower().startswith("paste") or re.fullmatch(r"token\d*", token.lower()):
            raise ValueError(f"APIFY_TOKEN entry #{position} is a placeholder; provide an Apify credential.")
        if token not in tokens:
            tokens.append(token)
    if not tokens:
        raise ValueError("APIFY_TOKEN must contain at least one non-empty token.")
    return tokens


def is_token_error(exc) -> bool:
    return getattr(exc, "status_code", None) in {401, 402, 403, 429}


def rotating_fetch(tokens, fetch_one=fetch, rng=random):
    live = list(enumerate(tokens, 1))
    rng.shuffle(live)
    total = len(live)
    index = 0

    def fetch_fn(queries, max_items):
        nonlocal index
        while live:
            position, token = live[index]
            print(f"[token #{position} of {total}]")
            try:
                items = fetch_one(queries, max_items, token)
            except Exception as exc:
                if not is_token_error(exc):
                    raise
                print(f"WARNING: token #{position} rejected (HTTP {exc.status_code}); "
                      "dropping it for this run")
                # ponytail: a 429 drops the token for the whole run; add cool-down if rate limits become common.
                live.pop(index)
                if live:
                    index %= len(live)
            else:
                index = (index + 1) % len(live)
                return items
        raise RuntimeError("All Apify tokens were rejected (auth, credit or rate limit). "
                           "Top up or wait, then rerun to resume.")

    return fetch_fn


def cost(n_items):
    return n_items * PRICE_PER_1000 / 1000


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]  # not splitlines(): it also splits on U+2028 inside tweets


def _append_jsonl(path, rows):
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _append_csv(path, fields, rows):
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def _minimized(items, salt, queries, seen=()):
    seen = set(seen)
    rows, skipped = [], 0
    for item in items:
        try:
            row = minimize(item, salt, queries)
        except (KeyError, TypeError, ValueError):
            # malformed item (actor schema drift): skip it, never crash after paying
            skipped += 1
            continue
        if row is not None and row["id"] not in seen:
            seen.add(row["id"])
            rows.append(row)
    if skipped:
        print(f"WARNING: skipped {skipped} malformed items; check the actor output schema")
    return rows


def _paid_fetch(fetch_fn, queries, max_items, confirm, label):
    message = f"{label}: at most {max_items} items; cost bound ${cost(max_items):.6f} USD"
    print(message)
    if not confirm(message):
        return None
    return fetch_fn(queries, max_items)


def final(data_dir, fetch_fn, salt, confirm):
    data_dir = Path(data_dir)
    seen = {row["id"] for row in _read_jsonl(data_dir / "final_posts.jsonl")}
    log_path = data_dir / "collection_log.csv"
    recorded = {(row["keyword"], row["month"]) for row in _read_csv(log_path)}
    windows = month_windows(START, END)
    for term in KEYWORDS:
        if all((term, a[:7]) in recorded for a, _ in windows):
            continue
        queries = build_queries(term, windows)
        items = _paid_fetch(fetch_fn, queries, MAX_PER_QUERY_FINAL * len(queries),
                            confirm, f"Final collection {term}")
        if items is None:
            return
        rows = _minimized(items, salt, queries, seen)
        _append_jsonl(data_dir / "final_posts.jsonl", rows)
        seen.update(row["id"] for row in rows)
        counts = Counter(item.get("searchTermIndex") for item in items)
        logs = [{"keyword": term, "month": a[:7], "fetched": counts[i],
                 "capped": counts[i] >= MAX_PER_QUERY_FINAL}
                for i, (a, _) in enumerate(windows) if (term, a[:7]) not in recorded]
        _append_csv(log_path, ["keyword", "month", "fetched", "capped"], logs)
        recorded.update((row["keyword"], row["month"]) for row in logs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Collect MBG posts using predefined keywords.")
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--smoke", action="store_true")
    commands.add_argument("--dry-run", action="store_true")
    commands.add_argument("--final", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    missing = [name for name in ("APIFY_TOKEN", "USER_HASH_SALT") if not os.environ.get(name)]
    if missing:
        parser.error("Missing required environment variables: " + ", ".join(missing))
    try:
        args.tokens = parse_tokens(os.environ["APIFY_TOKEN"])
    except ValueError as error:
        parser.error(str(error))
    args.salt = os.environ["USER_HASH_SALT"]
    return args


def main(argv=None):
    args = parse_args(argv)
    tokens, salt = args.tokens, args.salt
    if args.dry_run:
        n_windows = len(month_windows(START, END))
        n_queries = len(KEYWORDS) * n_windows
        print(f"Final collection queries: {n_queries} ({len(KEYWORDS)} keywords x {n_windows} months)")
        print(f"Final collection cost bound: ${cost(n_queries * MAX_PER_QUERY_FINAL):.6f} USD")
        return

    fetch_fn = rotating_fetch(tokens)

    def confirm(message):
        return args.yes or input("Continue? [y/N] ").strip().lower() == "y"

    if args.smoke:
        queries = ["MBG"]
        items = _paid_fetch(fetch_fn, queries, 20, confirm, "Smoke test")
        if items is not None:
            print(list(items[0]) if items else [])
            rows = _minimized(items, salt, queries)
            print(json.dumps(rows[0] if rows else None, ensure_ascii=False))
        return

    final(Path("data"), fetch_fn, salt, confirm)


if __name__ == "__main__":
    main()
