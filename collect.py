import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

import judge


ACTOR_ID = "kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest"  # id CJdippxWmn9uRfooo, the actor behind fixtures/sample_tweet.json
SEEDS = ["MBG", "makan bergizi gratis"]
# user-chosen final list (replaces snowball discovery); X search is case-insensitive, judge masking too
KEYWORDS = ["mbg", "makan bergizi gratis", "sppg", "satuan pelayanan pemenuhan gizi", "bgn",
            "badan gizi nasional", "embege"]
START, END = "2025-01-01", "2026-10-26"
MAX_PER_QUERY_DISCOVERY = 25  # per (term, month window); the actor's minimum per query is 20
PILOT_N = 100
BACKGROUND_N = 300
BACKGROUND_QUERY = "yang"
TOP_N_CANDIDATES = 25
MIN_DF = 5
MIN_LIFT = 3.0
MAX_ACCEPT_PER_ROUND = 5
MIN_PRECISION_NOVEL = 0.70
MIN_NOVEL_N = 20
REVIEW_MIN_PRECISION = 0.5
REVIEW_EXAMPLES = 3
STOP_GAIN = 0.05
MAX_ROUNDS = 4
ENCODER = "indobenchmark/indobert-base-p1"
MAX_SEQ = 128
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


def candidate_terms(text, known):
    stopwords = set("""
        yang dan di ke dari untuk dengan pada adalah ini itu juga dalam tidak
        ada akan sebagai karena oleh atau sudah saja lebih bisa jadi telah
        saat agar kalau jika masih hanya kita kami saya aku kamu mereka dia
        nya pun para sangat semua tentang tapi bukan lagi sampai buat tanpa
        serta tersebut harus sudah memang seperti antara setelah sebelum
        the and for with this that from are was were have has not but you
        your our they their its into about
        gak ga nggak enggak tak aja yg dgn udah banget bgt sih nih tuh kok dong
        deh ya kan lah mah loh lho emang kayak kaya biar jg krn karna utk dr sama
        kalo ntar bikin pada gitu gini gimana apa siapa kenapa mengapa bagaimana
        kapan mana yaitu yakni walau meski namun tetapi hingga just like get one
        can will out all
    """.split())
    known_phrases = [term.lower().lstrip("#").split() for term in known]
    text = re.sub(r"https?://\S+|@\w+", " ", text.lower())
    tokens = re.findall(r"#?\w+", text)
    valid = [len(word.lstrip("#")) >= 3 and not word.lstrip("#").isdigit()
             and word.lstrip("#") not in stopwords for word in tokens]
    terms = {word for word, keep in zip(tokens, valid) if keep}
    terms.update(f"{a} {b}" for a, b, keep_a, keep_b in
                 zip(tokens, tokens[1:], valid, valid[1:]) if keep_a and keep_b)
    candidates = set()
    for term in terms:
        if judge.mask_terms(term, known) != term:
            continue
        words = [word.lstrip("#") for word in term.split()]
        if any(len(phrase) > 1 and any(
            phrase[i:i + len(words)] == words
            for i in range(len(phrase) - len(words) + 1)
        ) for phrase in known_phrases):
            continue
        candidates.add(term)
    return candidates


def extract_candidates(rows, known, top_n, min_df, background=None, min_lift=MIN_LIFT):
    counts = Counter(term for row in rows for term in candidate_terms(row["text"], known))
    if background is None:
        return sorted(((term, df, None) for term, df in counts.items() if df >= min_df),
                      key=lambda item: (-item[1], item[0]))[:top_n]
    background_counts = Counter(term for row in background
                                for term in candidate_terms(row["text"], known))
    candidates = []
    for term, df in counts.items():
        if df < min_df:
            continue
        lift = (df / len(rows)) / ((background_counts[term] + 1) / (len(background) + 1))
        if lift >= min_lift:
            candidates.append((term, df, lift))
    return sorted(candidates, key=lambda item: (-item[2], -item[1], item[0]))[:top_n]


def decide(novel_n, precision):
    if novel_n < MIN_NOVEL_N:
        return "reject_redundant"
    if precision < MIN_PRECISION_NOVEL:
        return "reject_precision"
    return "accept"


def prescreen(novel_n, precision):
    if novel_n < MIN_NOVEL_N:
        return "reject_redundant"
    if precision < REVIEW_MIN_PRECISION:
        return "reject_precision"
    return "review"


def pilot_examples(pilot_rows, known, k=REVIEW_EXAMPLES, seed=0):
    texts = [row["text"] for row in pilot_rows
             if judge.mask_terms(row["text"], known) == row["text"]]
    return [text[:240] for text in random.Random(seed).sample(texts, min(k, len(texts)))]


def approve_terms(shortlist, input_fn=input, print_fn=print, title=None):
    shortlist = sorted(shortlist, key=lambda result: -result["precision_novel"])
    if title is None:
        round_no = shortlist[0].get("round") if shortlist else None
        title = f"Round {round_no} review" if round_no is not None else "Review"
    print_fn(f"{title}: {len(shortlist)} candidate terms. "
             "Would a post containing ONLY this term be about MBG?")
    for number, result in enumerate(shortlist, 1):
        lift = f"{result['lift']:.1f}" if result["lift"] is not None else ""
        print_fn(f"  {number}. {result['term']}   df={result['df']} lift={lift} "
                 f"novel={result['novel_n']} judge-pass={result['precision_novel']:.2f}")
        for example in result["examples"]:
            print_fn("       - " + json.dumps(example, ensure_ascii=False))
    while True:
        print_fn("Accept (numbers separated by commas; add your own terms after ';'; "
                 "empty = accept none):")
        line = input_fn().strip()
        if not line:
            return [], []
        numbers, _, extras = line.partition(";")
        try:
            selected = {int(number.strip()) for number in numbers.split(",")} if numbers.strip() else set()
            if any(number < 1 or number > len(shortlist) for number in selected):
                raise ValueError
        except ValueError:
            print_fn(f"Invalid selection; enter numbers from 1 to {len(shortlist)}.")
            continue
        accepted = [result["term"] for number, result in enumerate(shortlist, 1)
                    if number in selected]
        added = list(dict.fromkeys(term.strip() for term in extras.split(",") if term.strip()))
        return accepted, added


def pilot_stats(rows, known, term, cal, encode):
    texts = [row["text"] for row in rows
             if judge.mask_terms(row["text"], known) == row["text"]]
    if not texts:
        return 0, 0.0
    scores = judge.score(texts, cal, encode, list(known) + [term])
    return len(texts), float(np.mean(scores >= cal["tau"]))


def pick_accepted(results, k):
    accepted = [result for result in results if result["decision"] == "accept"]
    accepted.sort(key=lambda result: (-result["precision_novel"], -result["novel_n"]))
    return [result["term"] for result in accepted[:k]]


def gain(new_on_topic, prev_on_topic):
    return new_on_topic / prev_on_topic if prev_on_topic else float("inf")


def should_stop(round_no, gain_value, n_accepted=None):
    if round_no >= MAX_ROUNDS:
        return "max_rounds"
    if round_no >= 2 and gain_value < STOP_GAIN:
        return "low_gain"
    if n_accepted == 0:
        return "no_accepted"
    return None


def cost(n_items):
    return n_items * PRICE_PER_1000 / 1000


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]  # not splitlines(): it also splits on U+2028 inside tweets


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


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


def _score_rows(rows, cal, encode):
    scores = judge.score([row["text"] for row in rows], cal, encode, SEEDS)
    for row, value in zip(rows, scores):
        row["judge_score"] = float(value) if math.isfinite(value) else None
        row["on_topic"] = bool(math.isfinite(value) and value >= cal["tau"])


def _paid_fetch(fetch_fn, queries, max_items, confirm, label):
    message = f"{label}: at most {max_items} items; cost bound ${cost(max_items):.6f} USD"
    print(message)
    if not confirm(message):
        return None
    return fetch_fn(queries, max_items)


def _freeze(data_dir, state, reason, picked_by=None):
    state["stop_reason"] = reason
    frozen = {
        "keywords": state["keywords"],
        "encoder": ENCODER,
        "tau": state["judge"]["tau"],
        "params": {name: value for name, value in globals().items() if name.isupper()},
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    if picked_by is not None:
        frozen["picked_by"] = picked_by
    _write_json(data_dir / "keywords_final.json", frozen)
    _write_json(data_dir / "state.json", state)
    print(f"Stopped: {reason}")


def pick(data_dir, review=approve_terms, repick=False):
    data_dir = Path(data_dir)
    state_path = data_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if "judge" not in state:
        raise SystemExit("run discovery first")
    frozen_path = data_dir / "keywords_final.json"
    if frozen_path.exists() and not repick:
        if json.loads(frozen_path.read_text()).get("picked_by") == "human":
            print("Final keywords already picked; use --repick to choose again.")
            return
    if "pick" not in state or repick:
        latest = {}
        for row in _read_jsonl(data_dir / "review_log.jsonl"):
            term = row["term"]
            if term in SEEDS or prescreen(row["novel_n"], row["precision_novel"]) != "review":
                continue
            if term not in latest or row["round"] >= latest[term]["round"]:
                latest[term] = row
        shortlist = sorted(latest.values(), key=lambda row: -row["precision_novel"])
        accepted, added = review(shortlist, title="Final pick")
        state["pick"] = {"accepted": list(accepted), "added": list(added)}
    choice = state["pick"]
    state["keywords"] = list(dict.fromkeys(SEEDS + choice["accepted"] + choice["added"]))
    state.update(stop_reason="human_pick", pending=[])
    # Persist the answer before freezing so a resumed pick never asks twice.
    _write_json(state_path, state)
    _freeze(data_dir, state, "human_pick", picked_by="human")
    print("Final keywords: " + ", ".join(state["keywords"]))


def _collect_round(data_dir, state, rows, fetch_fn, encode, salt, confirm):
    round_no = state["round"] + 1
    progress = state.setdefault("collecting", {
        "prev_unique": len(rows), "prev_on_topic": sum(row["on_topic"] for row in rows)
    })
    state_path = data_dir / "state.json"
    _write_json(state_path, state)
    terms = state["pending"]
    queries = [q for term in terms for q in build_queries(term, month_windows(START, END))]
    # ponytail: a crash mid-round may re-collect that round; id dedupe is correct but costs money.
    if "seed_rows" not in progress:
        items = _paid_fetch(fetch_fn, queries, MAX_PER_QUERY_DISCOVERY * len(queries),
                            confirm, f"Discovery round {round_no}")
        if items is None:
            return False
        new_rows = _minimized(items, salt, queries, (row["id"] for row in rows))
        fetched = len(items)
        if round_no == 1:
            progress.update(seed_rows=new_rows, fetched=fetched)
            _write_json(state_path, state)
    else:
        new_rows, fetched = progress["seed_rows"], progress["fetched"]
        seen = {row["id"] for row in rows}
        new_rows = [row for row in new_rows if row["id"] not in seen]
    calibration_path = data_dir / "calibration.npz"
    if round_no == 1:
        background_path = data_dir / "background.jsonl"
        if not background_path.exists():
            background_queries = build_queries(BACKGROUND_QUERY, [(START, END)])
            items = _paid_fetch(fetch_fn, background_queries, BACKGROUND_N,
                                confirm, "Background calibration sample")
            if items is None:
                return False
            _append_jsonl(background_path, _minimized(items, salt, background_queries))
        if not calibration_path.exists():
            cal = judge.calibrate(
                [row["text"] for row in progress["seed_rows"]],
                [row["text"] for row in _read_jsonl(background_path)], encode, SEEDS
            )
            judge.save_calibration(cal, calibration_path)
        cal = judge.load_calibration(calibration_path)
        state["judge"] = {
            "mu_path": "calibration.npz", "centroid_path": "calibration.npz",
            "tau": cal["tau"], "auc_r_vs_b": cal["auc"], "b0": cal["b0"],
            "encoder": ENCODER,
        }
    else:
        cal = judge.load_calibration(calibration_path)
    _score_rows(new_rows, cal, encode)
    _append_jsonl(data_dir / "posts.jsonl", new_rows)
    rows.extend(new_rows)
    new_on_topic = sum(row["on_topic"] for row in rows) - progress["prev_on_topic"]
    state["history"].append({
        "round": round_no, "keywords": list(terms), "fetched": fetched,
        "new_unique": len(rows) - progress["prev_unique"], "new_on_topic": new_on_topic,
        "gain": gain(new_on_topic, progress["prev_on_topic"]),
    })
    state.update(round=round_no, pending=[])
    del state["collecting"]
    _write_json(state_path, state)
    return True


def _log_reviews(data_dir, results):
    log_path = data_dir / "review_log.jsonl"
    logged = {(row["round"], row["term"]) for row in _read_jsonl(log_path)}
    fields = ("round", "term", "df", "lift", "novel_n", "precision_novel",
              "decision", "examples", "timestamp")
    _append_jsonl(log_path, [
        {key: result[key] for key in fields}
        for result in results if (result["round"], result["term"]) not in logged
    ])


def _review_pilots(data_dir, state, cal, review):
    progress = state["pilots"]
    results = progress["results"]
    for result in results:
        result["decision"] = prescreen(result["novel_n"], result["precision_novel"])
    shortlist = sorted((result for result in results if result["decision"] == "review"),
                       key=lambda result: -result["precision_novel"])
    if "review" not in progress:
        progress["review"] = review(shortlist)
        _write_json(data_dir / "state.json", state)
    accepted, added = progress["review"]
    for result in shortlist:
        result["decision"] = "accept_human" if result["term"] in accepted else "reject_human"
    _log_reviews(data_dir, shortlist)
    results.extend({
        "round": state["round"], "term": term, "df": 0, "lift": None, "novel_n": 0,
        "precision_novel": 0, "tau": cal["tau"], "b0": cal["b0"],
        "decision": "human_added", "pilot_cost_usd": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    } for term in added)
    return list(accepted) + list(added)


def _pilot_round(data_dir, state, rows, fetch_fn, encode, salt, confirm, review):
    if "pilots" not in state:
        state["pilots"] = {
            "candidates": extract_candidates(
                rows, state["keywords"], TOP_N_CANDIDATES, MIN_DF,
                background=_read_jsonl(data_dir / "background.jsonl")
            ),
            "results": [],
        }
    progress = state["pilots"]
    state_path = data_dir / "state.json"
    _write_json(state_path, state)
    if progress["candidates"] and not progress.get("confirmed"):
        count = len(progress["candidates"])
        message = (f"Pilots: {count} candidates; at most {count * PILOT_N} items; "
                   f"cost bound ${cost(count * PILOT_N):.6f} USD")
        print(message)
        if not confirm(message):
            return False
        progress["confirmed"] = True
        _write_json(state_path, state)
    cal = judge.load_calibration(data_dir / "calibration.npz")
    finished = {result["term"] for result in progress["results"]}
    for term, df, lift in progress["candidates"]:
        if term in finished:
            continue
        queries = build_queries(term, [(START, END)])
        items = fetch_fn(queries, PILOT_N)
        pilot_rows = _minimized(items, salt, queries)
        novel_n, precision = pilot_stats(pilot_rows, state["keywords"], term, cal, encode)
        progress["results"].append({
            "round": state["round"], "term": term, "df": df, "lift": lift, "novel_n": novel_n,
            "precision_novel": precision, "tau": cal["tau"], "b0": cal["b0"],
            "decision": prescreen(novel_n, precision) if review is not None else decide(novel_n, precision),
            "examples": pilot_examples(pilot_rows, state["keywords"]),
            "pilot_cost_usd": cost(len(items)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        print(f"Pilot {term}: novel_n={novel_n} precision={precision:.2f}")
        _write_json(state_path, state)
    results = progress["results"]
    if review is None:
        accepted = pick_accepted(results, MAX_ACCEPT_PER_ROUND)
        for result in results:
            if result["decision"] == "accept" and result["term"] not in accepted:
                result["decision"] = "reject_cap"
        _log_reviews(data_dir, results)
    else:
        accepted = _review_pilots(data_dir, state, cal, review)
    path = data_dir / "decisions.csv"
    recorded = {(int(row["round"]), row["term"]) for row in _read_csv(path)}
    fields = ["round", "term", "df", "lift", "novel_n", "precision_novel", "tau",
              "b0", "decision", "pilot_cost_usd", "timestamp"]
    _append_csv(path, fields, [{key: row[key] for key in fields} for row in results
                              if (row["round"], row["term"]) not in recorded])
    state["keywords"].extend(accepted)
    state["pending"] = accepted
    del state["pilots"]
    reason = should_stop(state["round"], state["history"][-1]["gain"], len(accepted))
    if reason:
        _freeze(data_dir, state, reason)
    else:
        _write_json(data_dir / "state.json", state)
    return True


def discover(data_dir, fetch_fn, encode, salt, confirm=lambda msg: True, review=None):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    state_path = data_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "round": 0, "keywords": list(SEEDS), "pending": list(SEEDS),
        "history": [], "stop_reason": None,
    }
    if state["stop_reason"]:
        print(f"Stopped: {state['stop_reason']}")
    elif state.get("judge") and state["judge"]["encoder"] != ENCODER:
        raise SystemExit(f"data/ was calibrated with {state['judge']['encoder']}, "
                         f"not {ENCODER}; use a fresh data dir to switch encoders.")
    rows = _read_jsonl(data_dir / "posts.jsonl")
    while not state["stop_reason"]:
        if state["pending"]:
            if not _collect_round(data_dir, state, rows, fetch_fn, encode, salt, confirm):
                return
        reason = should_stop(state["round"], state["history"][-1]["gain"])
        if reason:
            _freeze(data_dir, state, reason)
            break
        if not _pilot_round(data_dir, state, rows, fetch_fn, encode, salt, confirm, review):
            return
    if review is None:
        print(f"Discovery finished ({state['stop_reason']}). "
              "Next: run collect.py --pick to choose the final keywords.")


def freeze_predefined(data_dir):
    state = json.loads((data_dir / "state.json").read_text())
    state.update(keywords=list(KEYWORDS), pending=[])
    state.pop("pilots", None)
    _freeze(data_dir, state, "predefined", picked_by="human")


def final(data_dir, fetch_fn, encode, salt, confirm, allow_auto=False, predefined=False):
    data_dir = Path(data_dir)
    if predefined:
        freeze_predefined(data_dir)
    frozen_path = data_dir / "keywords_final.json"
    if not frozen_path.exists():
        raise SystemExit("Final collection requires keywords_final.json; run discovery first.")
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("picked_by") != "human" and not allow_auto:
        raise SystemExit("keywords were chosen automatically; run --pick first (or pass --allow-auto)")
    if frozen["encoder"] != ENCODER:
        raise SystemExit(f"keywords were frozen with {frozen['encoder']}, not {ENCODER}.")
    cal = judge.load_calibration(data_dir / "calibration.npz")
    seen = {row["id"] for row in _read_jsonl(data_dir / "final_posts.jsonl")}
    log_path = data_dir / "collection_log.csv"
    recorded = {(row["keyword"], row["month"]) for row in _read_csv(log_path)}
    windows = month_windows(START, END)
    for term in frozen["keywords"]:
        if all((term, a[:7]) in recorded for a, _ in windows):
            continue
        queries = build_queries(term, windows)
        items = _paid_fetch(fetch_fn, queries, MAX_PER_QUERY_FINAL * len(queries),
                            confirm, f"Final collection {term}")
        if items is None:
            return
        rows = _minimized(items, salt, queries, seen)
        _score_rows(rows, cal, encode)
        _append_jsonl(data_dir / "final_posts.jsonl", rows)
        seen.update(row["id"] for row in rows)
        counts = Counter(item.get("searchTermIndex") for item in items)
        logs = [{"keyword": term, "month": a[:7], "fetched": counts[i],
                 "capped": counts[i] >= MAX_PER_QUERY_FINAL}
                for i, (a, _) in enumerate(windows) if (term, a[:7]) not in recorded]
        _append_csv(log_path, ["keyword", "month", "fetched", "capped"], logs)
        recorded.update((row["keyword"], row["month"]) for row in logs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Discover MBG keywords and collect posts.")
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument("--smoke", action="store_true")
    commands.add_argument("--dry-run", action="store_true")
    commands.add_argument("--final", action="store_true")
    commands.add_argument("--pick", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--overnight", action="store_true")
    parser.add_argument("--repick", action="store_true")
    parser.add_argument("--allow-auto", action="store_true")
    parser.add_argument("--predefined", action="store_true", help="with --final: use KEYWORDS, skip discovery")
    args = parser.parse_args(argv)
    if args.predefined and not args.final:
        parser.error("--predefined only works with --final")
    if args.max_rounds is not None and args.max_rounds < 1:
        parser.error("--max-rounds must be at least 1")
    if args.max_rounds is None:
        args.max_rounds = 3 if args.overnight else MAX_ROUNDS
    if args.overnight:
        args.auto = args.yes = True
    args.tokens, args.salt = [], None
    if args.pick:
        if not sys.stdin.isatty():
            parser.error("--pick needs a terminal")
        return args
    missing = [name for name in ("APIFY_TOKEN", "USER_HASH_SALT") if not os.environ.get(name)]
    if missing:
        parser.error("Missing required environment variables: " + ", ".join(missing))
    try:
        args.tokens = parse_tokens(os.environ["APIFY_TOKEN"])
    except ValueError as error:
        parser.error(str(error))
    args.salt = os.environ["USER_HASH_SALT"]
    if not (args.smoke or args.final or args.dry_run or args.auto) and not sys.stdin.isatty():
        parser.error("human review needs a terminal; run it in a terminal tab or pass --auto")
    return args


def main(argv=None):
    global MAX_ROUNDS
    args = parse_args(argv)
    MAX_ROUNDS = args.max_rounds
    if args.pick:
        pick(Path("data"), repick=args.repick)
        return
    tokens, salt = args.tokens, args.salt
    if args.dry_run:
        print(f"Tokens loaded: {len(tokens)}")
        queries = [q for term in SEEDS for q in build_queries(term, month_windows(START, END))]
        print(f"Round 1 queries: {len(queries)} ({len(SEEDS)} seeds x "
              f"{len(month_windows(START, END))} months)")
        print("\n".join(queries))
        discovery_n = len(queries) * MAX_PER_QUERY_DISCOVERY
        print(f"Round 1 discovery cost bound ({len(queries)} x {MAX_PER_QUERY_DISCOVERY} "
              f"= {discovery_n} items): ${cost(discovery_n):.6f} USD")
        print(f"Background ({BACKGROUND_N} items): "
              + build_queries(BACKGROUND_QUERY, [(START, END)])[0])
        print(f"Background cost bound: ${cost(BACKGROUND_N):.6f} USD")
        bound = discovery_n + BACKGROUND_N
        print(f"Round 1 collection cost bound: ${cost(bound):.6f} USD")
        final_n = len(KEYWORDS) * len(month_windows(START, END)) * MAX_PER_QUERY_FINAL
        print(f"Final collection ({len(KEYWORDS)} predefined keywords): at most {final_n} items, "
              f"${cost(final_n):.6f} USD")
        print(f"Including up to {TOP_N_CANDIDATES} pilots: "
              f"${cost(bound + TOP_N_CANDIDATES * PILOT_N):.6f} USD")
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

    encoder = None

    def encode(texts):
        nonlocal encoder
        if encoder is None:
            encoder = judge.load_encoder(ENCODER, MAX_SEQ)
        return encoder(texts)

    if args.final:
        final(Path("data"), fetch_fn, encode, salt, confirm, allow_auto=args.allow_auto,
              predefined=args.predefined)
    else:
        discover(Path("data"), fetch_fn, encode, salt, confirm,
                 review=None if args.auto else approve_terms)


if __name__ == "__main__":
    main()
