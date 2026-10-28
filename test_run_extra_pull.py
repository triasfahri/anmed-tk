import copy
import csv
import hashlib
import io
import json
import os
import socket
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import collect
import run_extra_pull as executor


TOKENS = ["apify_fake_private_alpha", "apify_fake_private_alpha_extended",
          "apify_fake_private_gamma", "apify_fake_private_delta"]
SALT = "offline-same-collection-salt"
FIXTURE = json.loads((Path(__file__).parent / "fixtures/sample_tweet.json").read_text(encoding="utf-8"))
START = datetime(2025, 1, 2, tzinfo=timezone.utc)
FIELDS = [
    "chunk", "keyword", "months", "token_index", "queries", "queries_hash",
    "items_returned", "kept_new", "dropped_lang_or_rt", "dropped_duplicate",
    "dropped_out_of_window", "excluded_term_hits", "malformed", "est_cost_usd",
    "started_at", "finished_at", "status", "error",
]


def unexpected_call(*args, **kwargs):
    assert False, "No real fetch, credit lookup, Apify client, or network call is allowed"


def raw_tweet(identifier, text="bgn berita", created=START, index=0, lang="in"):
    item = copy.deepcopy(FIXTURE)
    item.update(id=identifier, text=text, createdAt=created.strftime("%a %b %d %H:%M:%S %z %Y"),
                lang=lang, searchTermIndex=index)
    return item


def query(start=START, keyword="bgn"):
    term = f'"{keyword}"' if " " in keyword else keyword
    return (f'{term} -mbg -"makan bergizi gratis" -sppg '
            f"since_time:{int(start.timestamp())} until_time:{int((start + timedelta(hours=6)).timestamp())}")


def chunk(starts=(START,), keyword="bgn", index=1):
    queries = [query(start, keyword) for start in starts]
    cap = len(queries) * 20
    return {
        "keyword": keyword, "months": list(dict.fromkeys(start.strftime("%Y-%m") for start in starts)),
        "queries": queries, "actor_input": collect.build_actor_input(queries, cap),
        "token_index": index, "items_cap": cap, "cost_bound": cap * 0.00025,
        "safe_bound": float(Decimal(cap) * Decimal("0.000275")),
    }


def workspace(directory, chunks):
    root = Path(directory)
    plan_path, data_dir = root / "plan.json", root / "data"
    plan_path.write_text(json.dumps({"parameters": {}, "notes": [], "chunks": chunks}),
                         encoding="utf-8")
    data_dir.mkdir(exist_ok=True)
    return plan_path, data_dir


def run(plan_path, data_dir, fetch_fn=unexpected_call, credit_fn=lambda token: Decimal("10"),
        flags=(), raw_tokens=None):
    output, errors = io.StringIO(), io.StringIO()
    env = {"APIFY_TOKEN": raw_tokens or ",".join(TOKENS), "USER_HASH_SALT": SALT}
    with patch.dict(os.environ, env, clear=True), redirect_stdout(output), redirect_stderr(errors):
        result = executor.main(["--yes", "--plan", str(plan_path), *flags], fetch_fn=fetch_fn,
                               credit_fn=credit_fn, data_dir=data_dir)
    assert result == 0 and not errors.getvalue()
    visible = output.getvalue() + errors.getvalue()
    log_path = data_dir / "extra_pull_log.csv"
    if log_path.exists():
        visible += log_path.read_text(encoding="utf-8")
    assert all(token not in visible for token in TOKENS)
    assert SALT not in visible
    return output.getvalue()


def read_posts(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def read_log(data_dir):
    with (data_dir / "extra_pull_log.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == FIELDS
        return list(reader)


def assert_counts(log, **counts):
    for name, value in counts.items():
        assert int(log[name]) == value, (name, log[name], value)
    classified = ["kept_new", "dropped_lang_or_rt", "dropped_duplicate", "dropped_out_of_window", "malformed"]
    assert sum(int(log[name]) for name in classified) == int(log["items_returned"])
    assert float(log["est_cost_usd"]) == collect.cost(int(log["items_returned"]))
    started, finished = (datetime.fromisoformat(log[name]) for name in ("started_at", "finished_at"))
    assert started.utcoffset() == finished.utcoffset() == timedelta(0)
    assert started <= finished


def test_requires_yes():
    output = io.StringIO()
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
            patch.object(Path, "read_text", unexpected_call), \
            patch.object(collect, "parse_tokens", unexpected_call), redirect_stdout(output):
        result = executor.main(["--plan", "missing.json", "--chunks", "19"],
                               fetch_fn=unexpected_call, credit_fn=unexpected_call, data_dir=directory)
        assert result != 0 and not list(Path(directory).iterdir())
    assert "--yes is required" in output.getvalue() and "nothing spent" in output.getvalue()


def test_requires_environment():
    for env, missing in (({}, ("APIFY_TOKEN", "USER_HASH_SALT")),
                         ({"APIFY_TOKEN": TOKENS[0]}, ("USER_HASH_SALT",)),
                         ({"USER_HASH_SALT": SALT}, ("APIFY_TOKEN",))):
        errors = io.StringIO()
        with patch.dict(os.environ, env, clear=True), redirect_stderr(errors):
            try:
                executor.main(["--yes"], fetch_fn=unexpected_call, credit_fn=unexpected_call)
            except SystemExit as error:
                assert error.code == 2
            else:
                assert False, "Missing environment must refuse spending"
        assert all(name in errors.getvalue() for name in missing)
        assert all(token not in errors.getvalue() for token in TOKENS)
        assert SALT not in errors.getvalue()


def test_chunk_selection():
    assert executor.parse_chunks(None, 19) == list(range(1, 20))
    assert executor.parse_chunks("19", 19) == [19]
    assert executor.parse_chunks("1-3", 19) == [1, 2, 3]
    assert executor.parse_chunks("19, 1-3,2,5-6,6", 19) == [1, 2, 3, 5, 6, 19]
    assert executor.parse_chunks(" 19 , 1 - 3 ", 19) == [1, 2, 3, 19]
    for raw in ("", " ", "0", "20", "2-1", "1-20", "-1", "1-", "1,,2", "a", "1.0", "1-2-3"):
        try:
            executor.parse_chunks(raw, 19)
        except ValueError:
            pass
        else:
            assert False, "Invalid selection must refuse spending"


def test_dedupe_counts_fields_and_unicode():
    february = datetime(2025, 2, 2, 7, tzinfo=timezone.utc)
    chunks = [chunk((START, february), index=2),
              chunk(keyword="satuan pelayanan pemenuhan gizi")]
    with tempfile.TemporaryDirectory() as directory:
        plan_path, data_dir = workspace(directory, chunks)
        final_path, extra_path = data_dir / "final_posts.jsonl", data_dir / "extra_posts.jsonl"
        collect._append_jsonl(final_path, [collect.minimize(raw_tweet("in-final", "earlier\u2028final"), SALT)])
        collect._append_jsonl(extra_path, [collect.minimize(raw_tweet("in-extra", "earlier\u2028extra"), SALT)])
        final_before, extra_before, plan_before = final_path.read_bytes(), extra_path.read_bytes(), plan_path.read_bytes()
        retweet = raw_tweet("retweet")
        retweet["retweeted_tweet"] = {"id": "original"}
        missing_text = raw_tweet("bad-text")
        del missing_text["text"]
        bad_date = raw_tweet("bad-date")
        bad_date["createdAt"] = "not a date"
        bad_author = raw_tweet("bad-author")
        bad_author["author"] = {}
        bad_id = raw_tweet(["unhashable"])
        first_items = [
            raw_tweet("in-final"), raw_tweet("in-extra"),
            raw_tweet("new", "bgn baris\u2028kedua"), raw_tweet("new", "same item again"),
            raw_tweet("english", lang="en"), retweet,
            raw_tweet("later-valid", created=START - timedelta(seconds=1)),
            raw_tweet("at-until", created=START + timedelta(hours=6)),
            raw_tweet("hit-all", "MBG, mBg! Makan Bergizi Gratis dan SPPG."),
            raw_tweet("substrings", "ambg mbgx _sppg sppg_ xmakan bergizi gratisx"),
            raw_tweet("hit-phrase", "Makan Bergizi Gratis!"),
            raw_tweet("later-valid", created=START + timedelta(seconds=1)),
            raw_tweet("feb-post", "bgn februari", created=february, index=1, lang="id"),
            missing_text, bad_date,
            *(raw_tweet(f"bad-index-{index}", index=index) for index in (None, -1, 2, "0")),
            bad_author, None, bad_id,
        ]
        assert len(first_items) == 22
        fetch_calls, credits = [], []

        def fetch(queries, max_items, token):
            fetch_calls.append((queries, max_items, token))
            if len(fetch_calls) == 1:
                assert extra_path.read_bytes() == extra_before
                assert not (data_dir / "extra_pull_log.csv").exists()
                return first_items
            assert len(read_log(data_dir)) == 1
            assert len(read_posts(extra_path)) == 7
            return [raw_tweet("new"), raw_tweet("chunk2-new")]

        def credit(token):
            credits.append(token)
            return Decimal("10")

        output = run(plan_path, data_dir, fetch, credit)
        assert fetch_calls == [(chunks[0]["queries"], 40, TOKENS[1]),
                               (chunks[1]["queries"], 20, TOKENS[0])]
        assert credits == [TOKENS[1], TOKENS[0], TOKENS[0], TOKENS[1]]
        assert final_path.read_bytes() == final_before and plan_path.read_bytes() == plan_before
        assert extra_path.read_bytes().startswith(extra_before)
        posts = read_posts(extra_path)
        assert [row["id"] for row in posts] == [
            "in-extra", "new", "hit-all", "substrings", "hit-phrase", "later-valid", "feb-post", "chunk2-new"]
        by_id = {row["id"]: row for row in posts}
        assert by_id["new"]["text"] == "bgn baris\u2028kedua"
        assert b"\xe2\x80\xa8" in extra_path.read_bytes()[len(extra_before):]
        for row in posts[1:]:
            start = february if row["id"] == "feb-post" else START
            keyword = chunks[1]["keyword"] if row["id"] == "chunk2-new" else "bgn"
            assert row["pull_batch"] == "slot_sample"
            assert row["source_keyword"] == keyword
            assert row["source_month"] == start.strftime("%Y-%m")
            assert row["slot_since"] == start.isoformat()
            assert row["slot_until"] == (start + timedelta(hours=6)).isoformat()
            assert row["query"] == query(start, keyword)
            assert row["user_hash"] == collect.minimize(FIXTURE, SALT)["user_hash"]
        logs = read_log(data_dir)
        assert len(logs) == 2 and all(row["status"] == "ok" and row["error"] == "" for row in logs)
        assert_counts(logs[0], items_returned=22, kept_new=6, dropped_lang_or_rt=2,
                      dropped_duplicate=3, dropped_out_of_window=2, excluded_term_hits=2, malformed=9)
        assert_counts(logs[1], items_returned=2, kept_new=1, dropped_lang_or_rt=0,
                      dropped_duplicate=1, dropped_out_of_window=0, excluded_term_hits=0, malformed=0)
        for number, log in enumerate(logs, 1):
            expected = chunks[number - 1]
            assert log["chunk"] == str(number) and log["keyword"] == expected["keyword"]
            assert log["months"] == ";".join(expected["months"])
            assert log["token_index"] == str(expected["token_index"])
            assert int(log["queries"]) == len(expected["queries"])
            assert log["queries_hash"] == hashlib.sha256("\n".join(expected["queries"]).encode()).hexdigest()[:16]
        assert output.count("\nchunk ") == 2
        assert "total items returned 24; total kept 7; total estimated cost $0.006000" in output
        assert "Failed chunks: none" in output and "Skipped chunks: none" in output


def test_selection_order_budget_and_token_positions():
    indices = [1, 3, 4] * 6 + [1]
    chunks = [chunk((START + timedelta(days=i),), index=index) for i, index in enumerate(indices)]
    with tempfile.TemporaryDirectory() as directory:
        plan_path, data_dir = workspace(directory, chunks)
        fetch_calls, credits = [], []

        def credit(token):
            printed = sys.stdout.getvalue()
            assert "Selected chunks: 4; items cap 80; total cost bound $0.020000; total safe bound $0.022000" in printed
            assert "token #1: items cap 40; cost bound $0.010000; safe bound $0.011000" in printed
            assert "token #3: items cap 20; cost bound $0.005000; safe bound $0.005500" in printed
            assert "token #4: items cap 20; cost bound $0.005000; safe bound $0.005500" in printed
            assert "token #2:" not in printed
            credits.append(token)
            return 10

        def fetch(queries, max_items, token):
            fetch_calls.append((queries, max_items, token))
            return []

        raw = f" '{TOKENS[0]}', \"{TOKENS[1]}\", {TOKENS[2]}, {TOKENS[3]}, {TOKENS[0]} "
        output = run(plan_path, data_dir, fetch, credit, flags=("--chunks", "19,1-3,2"), raw_tokens=raw)
        selected = [1, 2, 3, 19]
        assert fetch_calls == [(chunks[i - 1]["queries"], 20, TOKENS[indices[i - 1] - 1]) for i in selected]
        assert [int(row["chunk"]) for row in read_log(data_dir)] == selected
        assert [line.split("/", 1)[0] for line in output.split("\n") if line.startswith("chunk ")] == [
            f"chunk {i}" for i in selected]
        assert credits == [TOKENS[0], TOKENS[2], TOKENS[3], TOKENS[0], TOKENS[0], TOKENS[2], TOKENS[3]]
        assert (data_dir / "extra_posts.jsonl").read_bytes() == b""
        assert not (data_dir / "final_posts.jsonl").exists()


def test_failure_continues_and_resume_uses_hash():
    starts = [START + timedelta(days=i) for i in range(3)]
    chunks = [chunk((start,), index=i + 1) for i, start in enumerate(starts)]
    with tempfile.TemporaryDirectory() as directory:
        plan_path, data_dir = workspace(directory, chunks)
        calls = []

        def fetch(queries, cap, token):
            calls.append(queries)
            number = next(i for i, c in enumerate(chunks) if c["queries"] == queries)
            if number == 1:
                raise RuntimeError(f"request {TOKENS[1]} with {TOKENS[0]}, {TOKENS[2]} and {TOKENS[3]}: " + "x" * 160)
            return [raw_tweet(f"new-{number}", created=starts[number])]

        output = run(plan_path, data_dir, fetch)
        assert calls == [c["queries"] for c in chunks]
        logs = read_log(data_dir)
        assert [row["status"] for row in logs] == ["ok", "failed", "ok"]
        assert logs[1]["error"].startswith("RuntimeError: request <token> with <token>, <token> and <token>:")
        assert len(logs[1]["error"]) == len("RuntimeError: ") + 120
        assert_counts(logs[1], items_returned=0, kept_new=0, malformed=0)
        assert "Summary: ok 2 failed 1 skipped 0" in output and "Failed chunks: 2" in output
        assert [row["id"] for row in read_posts(data_dir / "extra_posts.jsonl")] == ["new-0", "new-2"]

        retry_calls = []

        def retry(queries, cap, token):
            retry_calls.append((queries, cap, token))
            return [raw_tweet("retried", created=starts[1])]

        # Exercise the default dependency wiring with offline replacements, too.
        with patch.object(collect, "fetch", retry), patch.object(executor, "read_credit", lambda token: 10):
            output = run(plan_path, data_dir, fetch_fn=None, credit_fn=None)
        assert retry_calls == [(chunks[1]["queries"], 20, TOKENS[1])]
        assert "Summary: ok 1 failed 0 skipped 0 already_ok 2" in output
        logs = read_log(data_dir)
        assert len(logs) == 4 and [row["status"] for row in logs] == ["ok", "failed", "ok", "ok"]
        before = {p.name: p.read_bytes() for p in data_dir.iterdir()}
        output = run(plan_path, data_dir, credit_fn=unexpected_call)
        assert "already_ok 3" in output
        assert {p.name: p.read_bytes() for p in data_dir.iterdir()} == before

        # A matching chunk number alone cannot suppress changed queries.
        chunks[0] = chunk((START + timedelta(days=10),))
        plan_path.write_text(json.dumps({"parameters": {}, "notes": [], "chunks": chunks}), encoding="utf-8")
        changed_calls = []

        def changed(queries, cap, token):
            changed_calls.append(queries)
            return [raw_tweet("changed", created=START + timedelta(days=10))]

        run(plan_path, data_dir, changed)
        assert changed_calls == [chunks[0]["queries"]]
        logs = read_log(data_dir)
        assert logs[-1]["chunk"] == "1" and logs[-1]["queries_hash"] != logs[0]["queries_hash"]


def test_credit_guard_threshold_and_retry():
    for remaining in (Decimal("0.025499"), Decimal("0.0255"), 0.0255):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, data_dir = workspace(directory, [chunk()])
            calls = []

            def fetch(queries, cap, token):
                calls.append(token)
                return []

            output = run(plan_path, data_dir, fetch, lambda token: remaining)
            log = read_log(data_dir)[0]
            if remaining == Decimal("0.025499"):
                assert not calls and log["status"] == "skipped" and log["error"] == "credit"
                assert "Skipped chunks: 1" in output and "failed 0 skipped 1" in output
                assert not (data_dir / "extra_posts.jsonl").exists()
                run(plan_path, data_dir, fetch)
                assert calls == [TOKENS[0]] and read_log(data_dir)[-1]["status"] == "ok"
            else:
                assert calls == [TOKENS[0]] and log["status"] == "ok"
            assert_counts(log, items_returned=0, kept_new=0)


def test_credit_lookup_failure_is_closed_and_summary_unknown():
    for value in (None, Decimal("NaN"), Decimal("Infinity")):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, data_dir = workspace(directory, [chunk()])

            def credit(token):
                if value is None:
                    raise ConnectionError(f"Cannot read {TOKENS[0]} or {TOKENS[1]}")
                return value

            output = run(plan_path, data_dir, credit_fn=credit)
            log = read_log(data_dir)[0]
            assert log["status"] == "skipped" and log["error"].startswith("credit lookup ")
            assert "Remaining credit token #1: unknown" in output
            assert_counts(log, items_returned=0, kept_new=0)

    with tempfile.TemporaryDirectory() as directory:
        plan_path, data_dir = workspace(directory, [chunk()])
        calls = []

        def credit(token):
            calls.append(token)
            if len(calls) == 1:
                return 10
            raise RuntimeError(TOKENS[0])

        output = run(plan_path, data_dir, lambda *args: [], credit)
        assert "Remaining credit token #1: unknown" in output
        assert read_log(data_dir)[0]["status"] == "ok"


def test_error_redaction_at_truncation_boundary():
    with tempfile.TemporaryDirectory() as directory:
        plan_path, data_dir = workspace(directory, [chunk(), chunk(keyword="embege")])
        calls = []

        def fetch(*args):
            calls.append(args)
            if len(calls) == 1:
                raise ValueError("x" * 110 + TOKENS[1] + "\n" + TOKENS[0])
            return [raw_tweet("after-failure")]

        output = run(plan_path, data_dir, fetch)
        error = read_log(data_dir)[0]["error"]
        assert error.startswith("ValueError: " + "x" * 110 + "<token>")
        assert "apify_fake" not in output + error
        assert [row["status"] for row in read_log(data_dir)] == ["failed", "ok"]
        assert len([line for line in output.split("\n") if line.startswith("chunk ")]) == 2


def test_invalid_plan_refuses_all_selected_chunks():
    invalid = [
        ("token_index", None), ("token_index", 0), ("token_index", 5), ("token_index", True),
        ("items_cap", 19), ("items_cap", 20.0), ("cost_bound", 0), ("safe_bound", 0),
        ("safe_bound", float("nan")), ("safe_bound", float("inf")),
        ("queries", []), ("queries", ["bgn since:2025-01-02 until:2025-01-03"]),
        ("queries", ["bgn since_time:1735797600 until_time:1735776000"]),
    ]
    for field, value in invalid:
        chunks = [chunk(), chunk(keyword="embege")]
        chunks[1][field] = value
        with tempfile.TemporaryDirectory() as directory:
            plan_path, data_dir = workspace(directory, chunks)
            output, errors = io.StringIO(), io.StringIO()
            with patch.dict(os.environ, {"APIFY_TOKEN": ",".join(TOKENS), "USER_HASH_SALT": SALT}, clear=True), \
                    redirect_stdout(output), redirect_stderr(errors):
                try:
                    executor.main(["--yes", "--plan", str(plan_path)], fetch_fn=unexpected_call,
                                   credit_fn=unexpected_call, data_dir=data_dir)
                except SystemExit as error:
                    assert error.code == 2
                else:
                    assert False, "Invalid later chunks must be caught before any spending"
            assert not list(data_dir.iterdir()) and not output.getvalue()
            assert "Cannot execute plan" in errors.getvalue()
            assert all(token not in errors.getvalue() for token in TOKENS)


if __name__ == "__main__":
    with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"apify_client": None}), \
            patch.object(collect, "fetch", unexpected_call), patch.object(executor, "read_credit", unexpected_call), \
            patch.object(socket.socket, "connect", unexpected_call), \
            patch.object(socket.socket, "connect_ex", unexpected_call), \
            patch.object(socket, "create_connection", unexpected_call):
        for name, test in list(globals().items()):
            if name.startswith("test_") and callable(test):
                test()
    print("ok")
