import copy
import csv
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import collect


def sample_tweet():
    return json.loads((Path(__file__).parent / "fixtures/sample_tweet.json").read_text())


def raw_tweet(identifier, text, index=0, lang="in"):
    item = copy.deepcopy(sample_tweet())
    item.update(id=identifier, text=text, lang=lang, searchTermIndex=index)
    return item


def test_minimize():
    item = sample_tweet()
    row = collect.minimize(item, "salt", ["seed query"])
    assert set(row) == {
        "id", "text", "created_at", "likes", "retweets", "replies", "quotes",
        "type", "parent_id", "lang", "user_hash", "bio", "followers", "query",
    }
    assert row["type"] == "reply"
    assert row["parent_id"] == "2100182645535879648"
    assert row["created_at"] == "2026-09-18T04:29:45+00:00"
    assert row["bio"] == item["author"]["profile_bio"]["description"]
    assert row["followers"] == 1810
    assert row["query"] == "seed query"
    assert row["text"].startswith("@pengguna ")
    assert row["user_hash"] == hashlib.sha256(
        ("salt" + item["author"]["id"]).encode()
    ).hexdigest()[:16]
    assert row["user_hash"] == collect.minimize(item, "salt")["user_hash"]
    assert row["user_hash"] != collect.minimize(item, "other salt")["user_hash"]
    serialized = json.dumps(row)
    for private in ("myjaem13", "putt", "profile_images", "profile_banners", "GusDikaOfficial"):
        assert private not in serialized
    assert collect.minimize(item, "salt")["query"] is None
    for index in (None, -1, 1, "0"):
        item["searchTermIndex"] = index
        assert collect.minimize(item, "salt", ["query"])["query"] is None
    item["text"] = "@abc @DEF_123 @nama"
    item["author"]["profile_bio"]["description"] = ""
    item["author"]["description"] = "fallback"
    assert collect.minimize(item, "salt")["text"] == "@pengguna @pengguna @pengguna"
    assert collect.minimize(item, "salt")["bio"] == "fallback"
    item["author"]["description"] = ""
    assert collect.minimize(item, "salt")["bio"] == ""


def test_minimize_filters_and_quotes():
    item = sample_tweet()
    item["retweeted_tweet"] = {"id": "retweet"}
    assert collect.minimize(item, "salt") is None
    item["retweeted_tweet"] = None
    item["lang"] = "en"
    assert collect.minimize(item, "salt") is None
    item["lang"] = "id"
    for key, quote in (("quoted_tweet", {"id": "quoted"}),
                       ("quoted_tweet_results", {"result": {"rest_id": "quoted"}})):
        item[key] = quote
        row = collect.minimize(item, "salt")
        assert row["type"] == "quote"
        assert row["parent_id"] == "quoted"
        item[key] = None
    item["isReply"] = False
    row = collect.minimize(item, "salt")
    assert row["type"] == "original" and row["parent_id"] is None


def test_month_windows():
    assert collect.month_windows("2025-01-01", "2025-03-17") == [
        ("2025-01-01", "2025-02-01"),
        ("2025-02-01", "2025-03-01"),
        ("2025-03-01", "2025-03-17"),
    ]
    assert collect.month_windows("2024-12-15", "2025-02-01") == [
        ("2024-12-15", "2025-01-01"), ("2025-01-01", "2025-02-01")
    ]
    assert collect.month_windows("2025-01-01", "2025-01-01") == []


def test_queries_and_actor_input():
    windows = [("2025-01-01", "2025-02-01"), ("2025-02-01", "2025-03-01")]
    queries = collect.build_queries("makan bergizi gratis", windows)
    assert queries == [
        '"makan bergizi gratis" since:2025-01-01_00:00:00_UTC until:2025-02-01_00:00:00_UTC',
        '"makan bergizi gratis" since:2025-02-01_00:00:00_UTC until:2025-03-01_00:00:00_UTC',
    ]
    assert collect.build_queries("MBG", windows)[0] == (
        "MBG since:2025-01-01_00:00:00_UTC until:2025-02-01_00:00:00_UTC"
    )
    # maxItems is per search term in this actor: the total is split across queries, floor 20
    assert collect.build_actor_input(queries, 200) == {
        "searchTerms": queries, "maxItems": 100, "queryType": "Latest", "lang": "in",
    }
    assert collect.build_actor_input(["q"], 5)["maxItems"] == 20
    assert collect.build_actor_input(["a", "b", "c"], 7)["maxItems"] == 20
    assert collect.build_actor_input(["a", "b", "c"], 3000)["maxItems"] == 1000


def test_fetch_uses_default_dataset():
    calls = []

    def actor(identifier):
        calls.append(identifier)

        def call(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(status="SUCCEEDED", status_message=None,
                                   default_dataset_id="dataset")

        return SimpleNamespace(call=call)

    def dataset(identifier):
        calls.append(identifier)
        return SimpleNamespace(iterate_items=lambda: iter([{"id": "one"}]))

    def client(token):
        calls.append(token)
        return SimpleNamespace(actor=actor, dataset=dataset)

    original = sys.modules.get("apify_client")
    sys.modules["apify_client"] = SimpleNamespace(ApifyClient=client)
    try:
        assert collect.fetch(["query"], 7, "secret") == [{"id": "one"}]
        assert calls == ["secret", collect.ACTOR_ID,
                         {"run_input": collect.build_actor_input(["query"], 7),
                          "max_items": 7, "logger": None}, "dataset"]
    finally:
        if original is None:
            del sys.modules["apify_client"]
        else:
            sys.modules["apify_client"] = original


def test_fetch_rejects_unsuccessful_runs():
    for status in ("FAILED", "ABORTED", "TIMED-OUT", "None"):
        run = None if status == "None" else SimpleNamespace(
            status=status, status_message="run stopped", default_dataset_id="unused"
        )
        client = SimpleNamespace(
            actor=lambda identifier: SimpleNamespace(call=lambda **kwargs: run),
            dataset=unexpected_call,
        )
        with patch.dict(sys.modules, apify_client=SimpleNamespace(ApifyClient=lambda token: client)):
            try:
                collect.fetch(["query"], 7, "secret")
            except RuntimeError as error:
                message = None if run is None else run.status_message
                assert str(error) == f"Apify run {status}: {message}"
            else:
                assert False, "only succeeded runs may read the dataset"


def test_fetch_keeps_items_of_cost_capped_run():
    run = SimpleNamespace(status="ABORTED", default_dataset_id="d",
                          status_message="This run was aborted because it reached its maximum cost of $0.03.")
    client = SimpleNamespace(
        actor=lambda identifier: SimpleNamespace(call=lambda **kwargs: run),
        dataset=lambda identifier: SimpleNamespace(iterate_items=lambda: iter([{"id": "one"}])),
    )
    with patch.dict(sys.modules, apify_client=SimpleNamespace(ApifyClient=lambda token: client)):
        assert collect.fetch(["query"], 7, "secret") == [{"id": "one"}]


def test_apify_run_model_fields():
    try:
        from apify_client._models import Run
    except ImportError:
        return
    assert {"status", "status_message", "default_dataset_id"} <= Run.model_fields.keys()


def test_parse_tokens():
    secret = "apify_api_private_alpha"
    assert collect.parse_tokens(f" '{secret}' , \"private-beta\", {secret}, ,") == [
        secret, "private-beta"
    ]
    assert collect.parse_tokens(secret) == [secret]
    assert collect.parse_tokens(f'  " {secret} "  ') == [secret]
    for raw in ("", " , '' , \"\" , "):
        try:
            collect.parse_tokens(raw)
        except ValueError as error:
            assert "at least one" in str(error)
            assert secret not in str(error)
        else:
            assert False, "empty token lists must be rejected"
    for placeholder in ("PASTE_YOUR_TOKEN_HERE", "paste-this", "token", "token1", "Token2"):
        try:
            collect.parse_tokens(f"{secret}, '{placeholder}',")
        except ValueError as error:
            assert "#2" in str(error) and "placeholder" in str(error)
            assert secret not in str(error) and placeholder not in str(error)
        else:
            assert False, "placeholder tokens must be rejected"


def test_is_token_error():
    assert not collect.is_token_error(RuntimeError("failed run"))
    for code in (401, 402, 403, 429, 400, 404, 500, None, "401"):
        error = Exception("request failed")
        error.status_code = code
        assert collect.is_token_error(error) == (code in {401, 402, 403, 429})


def test_rotating_fetch_round_robin():
    tokens = ["private-alpha", "private-beta", "private-gamma"]
    original = list(tokens)
    order = list(tokens)
    random.Random(0).shuffle(order)
    calls = []
    output = io.StringIO()

    def fetch_one(queries, max_items, token):
        assert queries == ["query"] and max_items == 7
        calls.append(token)
        return [{"id": str(len(calls))}]

    with redirect_stdout(output):
        fetch_fn = collect.rotating_fetch(tokens, fetch_one, random.Random(0))
        for i in range(6):
            assert fetch_fn(["query"], 7) == [{"id": str(i + 1)}]
    assert tokens == original
    assert calls == order * 2
    assert all(calls.count(token) == 2 for token in tokens)
    assert output.getvalue().splitlines() == [
        f"[token #{tokens.index(token) + 1} of 3]" for token in calls
    ]
    assert all(token not in output.getvalue() for token in tokens)


def test_rotating_fetch_drops_rejected_token():
    tokens = ["private-alpha", "private-beta", "private-gamma"]
    for rejected in tokens:
        calls = []
        output = io.StringIO()
        error = Exception("credit exhausted")
        error.status_code = 402

        def fetch_one(queries, max_items, token):
            calls.append(token)
            if token == rejected:
                raise error
            return [{"id": "ok"}]

        with redirect_stdout(output):
            fetch_fn = collect.rotating_fetch(tokens, fetch_one, random.Random(0))
            for _ in range(6):
                assert fetch_fn(["query"], 7) == [{"id": "ok"}]
        assert calls.count(rejected) == 1 and len(calls) == 7
        assert calls[calls.index(rejected) + 1] != rejected
        survivors = [token for token in calls if token != rejected]
        assert survivors[-4:] == survivors[-2:] * 2
        position = tokens.index(rejected) + 1
        assert (f"WARNING: token #{position} rejected (HTTP 402); "
                "dropping it for this run") in output.getvalue().splitlines()
        assert all(token not in output.getvalue() + str(error) for token in tokens)


def test_rotating_fetch_preserves_other_errors_and_tokens():
    tokens = ["private-alpha", "private-beta", "private-gamma"]
    server_error = Exception("server unavailable")
    server_error.status_code = 500
    for failure in (RuntimeError("Apify run FAILED: stopped"), server_error,
                    ConnectionError("network unavailable")):
        calls = []
        output = io.StringIO()

        def fetch_one(queries, max_items, token):
            calls.append(token)
            if len(calls) == 1:
                raise failure
            return [{"id": "ok"}]

        with redirect_stdout(output):
            fetch_fn = collect.rotating_fetch(tokens, fetch_one, random.Random(0))
            try:
                fetch_fn(["query"], 7)
            except Exception as error:
                assert error is failure
                assert all(token not in str(error) for token in tokens)
            else:
                assert False, "non-token errors must propagate unchanged"
            assert fetch_fn(["query"], 7) == [{"id": "ok"}]
            assert fetch_fn(["query"], 7) == [{"id": "ok"}]
        assert calls[0] == calls[1] and calls[2] != calls[1]
        assert "WARNING" not in output.getvalue()
        assert all(token not in output.getvalue() for token in tokens)


def test_rotating_fetch_all_tokens_rejected():
    tokens = ["private-alpha", "private-beta", "private-gamma"]
    codes = dict(zip(tokens, (401, 403, 429)))
    calls = []
    output = io.StringIO()

    def fetch_one(queries, max_items, token):
        calls.append(token)
        error = Exception("request rejected")
        error.status_code = codes[token]
        raise error

    with redirect_stdout(output):
        fetch_fn = collect.rotating_fetch(tokens, fetch_one, random.Random(0))
        for _ in range(2):
            try:
                fetch_fn(["query"], 7)
            except RuntimeError as error:
                assert str(error) == (
                    "All Apify tokens were rejected (auth, credit or rate limit). "
                    "Top up or wait, then rerun to resume."
                )
                assert all(token not in str(error) for token in tokens)
            else:
                assert False, "exhausting all tokens must raise"
    assert len(calls) == len(tokens) and set(calls) == set(tokens)
    for position, token in enumerate(tokens, 1):
        assert (f"WARNING: token #{position} rejected (HTTP {codes[token]}); "
                "dropping it for this run") in output.getvalue().splitlines()
    assert all(token not in output.getvalue() for token in tokens)


def test_cost():
    assert collect.cost(1000) == collect.PRICE_PER_1000 == 0.25
    assert collect.cost(0) == 0.0


def test_dry_run_and_missing_secrets():
    script = str(Path(__file__).with_name("collect.py"))
    env = dict(os.environ, APIFY_TOKEN="dummy-token", USER_HASH_SALT="dummy-salt")
    result = subprocess.run([sys.executable, script, "--dry-run"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "Final collection queries: 154 (7 keywords x 22 months)",
        "Final collection cost bound: $11.550000 USD",
    ]
    assert "dummy-token" not in result.stdout + result.stderr
    assert "dummy-salt" not in result.stdout + result.stderr
    env.pop("APIFY_TOKEN")
    env.pop("USER_HASH_SALT")
    result = subprocess.run([sys.executable, script, "--dry-run"], env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "APIFY_TOKEN" in result.stderr and "USER_HASH_SALT" in result.stderr


def test_dry_run_token_list_and_placeholder():
    for raw, valid in (("a1,b2,c3", True), ("PASTE_YOUR_TOKEN_HERE", False)):
        output, errors = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, APIFY_TOKEN=raw, USER_HASH_SALT="private-salt"), \
                patch.object(collect, "rotating_fetch", unexpected_call), \
                redirect_stdout(output), redirect_stderr(errors):
            try:
                collect.main(["--dry-run"])
            except SystemExit as error:
                assert not valid and error.code != 0
                assert raw not in str(error)
            else:
                assert valid
                assert "Final collection queries: 154" in output.getvalue()
        assert all(token not in output.getvalue() + errors.getvalue() for token in raw.split(","))
        assert "private-salt" not in output.getvalue() + errors.getvalue()
        if not valid:
            assert "placeholder" in errors.getvalue()


def test_parse_args():
    with patch.dict(os.environ, APIFY_TOKEN="a1,b2", USER_HASH_SALT="y"):
        for mode in ("--final", "--smoke", "--dry-run"):
            args = collect.parse_args([mode, "--yes"])
            assert getattr(args, mode[2:].replace("-", "_")) and args.yes
            assert args.tokens == ["a1", "b2"] and args.salt == "y"
        for flags in ([], ["--final", "--smoke"], ["--final", "--dry-run"],
                      ["--smoke", "--dry-run"], ["--pick"], ["--repick"],
                      ["--auto"], ["--allow-auto"], ["--max-rounds", "2"],
                      ["--overnight"], ["--predefined"], ["--final", "--predefined"]):
            with redirect_stderr(io.StringIO()):
                try:
                    collect.parse_args(flags)
                except SystemExit as error:
                    assert error.code == 2
                else:
                    assert False, "invalid flags or combined modes must fail"


def test_cli_final():
    with patch.dict(os.environ, APIFY_TOKEN="a1,b2", USER_HASH_SALT="y"), \
            patch.object(collect, "rotating_fetch", return_value=unexpected_call) as rotate, \
            patch.object(collect, "final") as final_fn:
        collect.main(["--final", "--yes"])
        rotate.assert_called_once_with(["a1", "b2"])
        assert final_fn.call_args.args[:3] == (Path("data"), unexpected_call, "y")
        assert not final_fn.call_args.kwargs
        with patch("builtins.input", unexpected_call):
            assert final_fn.call_args.args[3]("cost bound") is True
        collect.main(["--final"])
        confirm = final_fn.call_args.args[3]
        for answer, expected in ((" y ", True), ("N", False), ("", False)):
            with patch("builtins.input", return_value=answer) as ask:
                assert confirm("cost bound") is expected
                ask.assert_called_once_with("Continue? [y/N] ")


def test_read_jsonl_survives_unicode_line_separator():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.jsonl"
        collect._append_jsonl(path, [{"id": "1", "text": "a\u2028b"}, {"id": "2", "text": "c"}])
        assert [row["id"] for row in collect._read_jsonl(path)] == ["1", "2"]


def file_snapshot(directory):
    return {path.name: path.read_bytes() for path in Path(directory).iterdir() if path.is_file()}


def unexpected_call(*args):
    assert False, "unexpected fetch or confirmation"


def test_smoke_cost_first():
    output = io.StringIO()
    item = sample_tweet()
    with patch.dict(os.environ, APIFY_TOKEN="secret-token", USER_HASH_SALT="secret-salt"), \
            patch.object(collect, "fetch", return_value=[item]) as mocked_fetch, \
            patch.object(collect, "rotating_fetch", return_value=collect.rotating_fetch(
                ["secret-token"], fetch_one=mocked_fetch, rng=random.Random(0)
            )) as mocked_rotation, \
            patch("builtins.input", unexpected_call), redirect_stdout(output):
        collect.main(["--smoke", "--yes"])
        mocked_rotation.assert_called_once_with(["secret-token"])
        mocked_fetch.assert_called_once_with(["MBG"], 20, "secret-token")
    lines = output.getvalue().splitlines()
    assert lines[0].startswith("Smoke test:") and "$0.005000" in lines[0]
    assert lines[1] == "[token #1 of 1]"
    assert "author" in lines[2] and "searchTermIndex" in lines[2]
    row = json.loads(lines[3])
    assert row == collect.minimize(item, "secret-salt", ["MBG"])
    assert "secret-token" not in output.getvalue() and "secret-salt" not in output.getvalue()


def test_predefined_keywords():
    assert collect.KEYWORDS == ["mbg", "makan bergizi gratis", "sppg", "satuan pelayanan pemenuhan gizi",
                                "bgn", "badan gizi nasional", "embege"]
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(collect, "START", "2025-01-01"), \
            patch.object(collect, "END", "2025-01-31"):
        terms = []

        def fetch_fn(queries, max_items):
            assert len(queries) == 1 and max_items == collect.MAX_PER_QUERY_FINAL
            terms.append(queries[0].split(" since:")[0])
            return []

        with redirect_stdout(io.StringIO()):
            collect.final(directory, fetch_fn, "salt", lambda message: True)
        assert terms == ["mbg", '"makan bergizi gratis"', "sppg", '"satuan pelayanan pemenuhan gizi"', "bgn",
                         '"badan gizi nasional"', "embege"]
        assert set(file_snapshot(directory)) == {"final_posts.jsonl", "collection_log.csv"}


def test_final_collection():
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(collect, "START", "2025-01-01"), \
            patch.object(collect, "END", "2025-03-17"), \
            patch.object(collect, "MAX_PER_QUERY_FINAL", 2):
        data_dir = Path(directory)
        calls = []

        def fetch_fn(queries, max_items):
            calls.append((queries, max_items))
            assert len(queries) == 3 and max_items == 6
            term = queries[0].split(" since:")[0]
            return [raw_tweet("shared", "post shared across keywords", 0),
                    raw_tweet(term + "-0", "monthly post", 0),
                    raw_tweet(term + "-1", "unrelated post", 1),
                    raw_tweet(term + "-2", "MBG", 2)]

        with redirect_stdout(io.StringIO()):
            collect.final(directory, fetch_fn, "salt", lambda message: True)
        assert len(calls) == len(collect.KEYWORDS)
        windows = collect.month_windows(collect.START, collect.END)
        assert [queries for queries, _ in calls] == [
            collect.build_queries(term, windows) for term in collect.KEYWORDS
        ]
        rows = collect._read_jsonl(data_dir / "final_posts.jsonl")
        assert len(rows) == len({row["id"] for row in rows}) == 1 + 3 * len(collect.KEYWORDS)
        assert all(set(row) == set(collect.minimize(sample_tweet(), "salt")) for row in rows)
        assert any(row["text"] == "unrelated post" for row in rows)
        assert all("since:" in row["query"] for row in rows)
        with (data_dir / "collection_log.csv").open() as stream:
            reader = csv.DictReader(stream)
            logs = list(reader)
            assert reader.fieldnames == ["keyword", "month", "fetched", "capped"]
        assert len(logs) == 3 * len(collect.KEYWORDS)
        for log in logs:
            assert int(log["fetched"]) == (2 if log["month"] == "2025-01" else 1)
            assert log["capped"] == ("True" if log["month"] == "2025-01" else "False")
        snapshot = file_snapshot(directory)
        with redirect_stdout(io.StringIO()):
            collect.final(directory, unexpected_call, "salt", unexpected_call)
        assert file_snapshot(directory) == snapshot


def test_final_partial_resume_and_declined_fetch():
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(collect, "START", "2025-01-01"), \
            patch.object(collect, "END", "2025-03-01"), \
            patch.object(collect, "KEYWORDS", ["mbg", "sppg"]):
        data_dir = Path(directory)
        posts_path = data_dir / "final_posts.jsonl"
        log_path = data_dir / "collection_log.csv"
        old_row = collect.minimize(raw_tweet("existing", "existing post"), "salt")
        collect._append_jsonl(posts_path, [old_row])
        old_logs = [
            {"keyword": "mbg", "month": "2025-01", "fetched": 9, "capped": False},
            {"keyword": "mbg", "month": "2025-02", "fetched": 8, "capped": False},
            {"keyword": "sppg", "month": "2025-01", "fetched": 7, "capped": False},
        ]
        collect._append_csv(log_path, ["keyword", "month", "fetched", "capped"], old_logs)
        snapshot = file_snapshot(directory)
        with redirect_stdout(io.StringIO()):
            collect.final(directory, unexpected_call, "salt", lambda message: False)
        assert file_snapshot(directory) == snapshot
        calls = []

        def fetch_fn(queries, max_items):
            calls.append((queries, max_items))
            return [raw_tweet("existing", "duplicate", 0),
                    raw_tweet("new", "new post\u2028with line separator", 1),
                    raw_tweet("new", "duplicate within batch", 1),
                    raw_tweet("english", "filtered", 1, lang="en"), {"lang": "in"}]

        output = io.StringIO()
        with redirect_stdout(output):
            collect.final(directory, fetch_fn, "salt", lambda message: True)
        windows = collect.month_windows(collect.START, collect.END)
        assert calls == [(collect.build_queries("sppg", windows), 2 * collect.MAX_PER_QUERY_FINAL)]
        assert "WARNING: skipped 1 malformed items" in output.getvalue()
        rows = collect._read_jsonl(posts_path)
        assert [row["id"] for row in rows] == ["existing", "new"] and rows[0] == old_row
        logs = collect._read_csv(log_path)
        assert logs[:3] == [{key: str(value) for key, value in row.items()} for row in old_logs]
        assert logs[3:] == [{"keyword": "sppg", "month": "2025-02", "fetched": "3", "capped": "False"}]
        assert posts_path.read_bytes().startswith(snapshot["final_posts.jsonl"])
        assert log_path.read_bytes().startswith(snapshot["collection_log.csv"])


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
    print("ok")
