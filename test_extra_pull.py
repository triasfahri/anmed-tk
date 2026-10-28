import csv
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, timedelta
from decimal import Decimal
from math import floor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import extra_pull as planner


def unexpected_call(*args, **kwargs):
    assert False, "No network, model loading, or fetching is allowed"


def fake_rows(keywords=("mbg", "sppg", "bgn")):
    return [{"keyword": keyword, "month": "2025-01", "capped": "True"}
            for keyword in keywords]


def test_sample_days():
    for length in (28, 30, 31):
        for requested in (1, 6, 14, length + 5):
            days = planner.sample_days(length, requested)
            count = min(requested, length)
            assert days == [1 + floor((i + 0.5) * length / count) for i in range(count)]
            assert len(days) == len(set(days)) == count
            assert all(1 <= day <= length for day in days)
            gaps = [b - a for a, b in zip(days, days[1:])]
            assert not gaps or max(gaps) - min(gaps) <= 1
            if requested > length:
                assert days == list(range(1, length + 1))
    assert planner.sample_days(28, 6) == [3, 8, 12, 17, 22, 26]
    assert planner.sample_days(30, 6) == [3, 8, 13, 18, 23, 28]
    assert planner.sample_days(31, 6) == [3, 8, 13, 19, 24, 29]
    for length, requested in ((0, 1), (28, 0), (31, -1)):
        try:
            planner.sample_days(length, requested)
        except ValueError:
            pass
        else:
            assert False, "Nonpositive sampling arguments must be rejected"


def test_query_formatting():
    assert planner.build_query("bgn", ["mbg", "makan bergizi gratis", "sppg"],
                               date(2025, 1, 31)) == (
        'bgn -mbg -"makan bergizi gratis" -sppg '
        "since:2025-01-31_00:00:00_UTC until:2025-02-01_00:00:00_UTC"
    )
    assert planner.build_query("makan bergizi gratis", [], date(2025, 12, 31)) == (
        '"makan bergizi gratis" since:2025-12-31_00:00:00_UTC '
        "until:2026-01-01_00:00:00_UTC"
    )
    assert planner.build_query("mbg", [], date(2024, 2, 29)) == (
        "mbg since:2024-02-29_00:00:00_UTC until:2024-03-01_00:00:00_UTC"
    )


def test_order_skips_and_exclusions():
    rows = [
        {"keyword": "sppg", "month": "2025-02", "capped": "False"},
        {"keyword": "mbg", "month": "2025-02", "capped": "True"},
        {"keyword": "sppg", "month": "2025-01", "capped": "True"},
        {"keyword": "mbg", "month": "2025-01", "capped": "True"},
        {"keyword": "makan bergizi gratis", "month": "2025-01", "capped": "True"},
        {"keyword": "bgn", "month": "2025-01", "capped": "False"},
        {"keyword": "embege", "month": "2025-01", "capped": "True"},
    ]
    assert planner.keyword_order(rows) == ["sppg", "mbg", "makan bergizi gratis", "bgn", "embege"]
    cells = planner.plan_cells(rows, 6)
    assert [(cell["keyword"], cell["month"]) for cell in cells] == [
        ("sppg", "2025-01"), ("mbg", "2025-01"), ("mbg", "2025-02"),
        ("makan bergizi gratis", "2025-01"), ("embege", "2025-01"),
    ]
    assert cells[2]["sampled_days"] == [3, 8, 12, 17, 22, 26]
    assert cells[3]["queries"][0].startswith('"makan bergizi gratis" -sppg -mbg since:')
    assert cells[4]["queries"][0].startswith('embege -sppg -mbg -"makan bergizi gratis" -bgn since:')
    queries = [query for cell in cells for query in cell["queries"]]
    assert len(queries) == len(set(queries)) == 30
    for query in queries:
        since, until = query.split(" since:")[1].split(" until:")
        assert since.endswith("_00:00:00_UTC") and until.endswith("_00:00:00_UTC")
        assert date.fromisoformat(until[:10]) - date.fromisoformat(since[:10]) == timedelta(days=1)
    leap = planner.plan_cells([{"keyword": "mbg", "month": "2024-02", "capped": "True"}], 40)
    assert leap[0]["sampled_days"] == list(range(1, 30))


def test_query_self_checks():
    for bad_query in (
        "mbg since:2025-01-01_01:00:00_UTC until:2025-01-02_00:00:00_UTC",
        "mbg since:2025-01-01_00:00:00_UTC until:2025-01-02_01:00:00_UTC",
        "mbg since:2025-01-01_00:00:00_UTC until:2025-01-03_00:00:00_UTC",
    ):
        with patch.object(planner, "build_query", return_value=bad_query):
            try:
                planner.plan_cells(fake_rows(("mbg",)), 1)
            except AssertionError:
                pass
            else:
                assert False, "Whole-day self-checks must reject invalid bounds"
    try:
        planner.plan_cells(fake_rows(("mbg", "mbg")), 1)
    except AssertionError:
        pass
    else:
        assert False, "Duplicate queries must be rejected"


def test_items_per_day():
    cells = planner.plan_cells(fake_rows(), 6)
    assert planner.ITEMS_PER_DAY == 40
    for cap in (20, 40, 41):
        chunks = planner.make_chunks(cells, cap)
        assert all(chunk["actor_input"]["maxItems"] == cap for chunk in chunks)
        assert all(chunk["items_cap"] == 6 * cap for chunk in chunks)
        assert all(chunk["cost_bound"] == 6 * cap * planner.PRICE for chunk in chunks)
        assert all(chunk["safe_bound"] == 6 * cap * planner.PRICE * planner.OVERSHOOT
                   for chunk in chunks)
    for cap in (0, 19):
        try:
            planner.make_chunks(cells, cap)
        except ValueError:
            pass
        else:
            assert False, "The actor's 20-item minimum must be enforced"
    with patch.object(planner, "read_remaining", unexpected_call), redirect_stderr(io.StringIO()):
        try:
            planner.main(["--items-per-day", "19"])
        except SystemExit as error:
            assert error.code == 2
        else:
            assert False, "CLI must reject fewer than 20 items before reading credit"


def test_chunking():
    rows = [{"keyword": "mbg", "month": f"2025-{month:02}", "capped": "True"}
            for month in range(8, 0, -1)]
    cells = planner.plan_cells(rows, 6)
    chunks = planner.make_chunks(cells)
    assert [len(chunk["months"]) for chunk in chunks] == [6, 2]
    assert chunks[0]["months"] == [f"2025-{month:02}" for month in range(1, 7)]
    assert chunks[1]["months"] == ["2025-07", "2025-08"]
    assert [len(chunk["queries"]) for chunk in chunks] == [36, 12]
    assert [chunk["items_cap"] for chunk in chunks] == [1440, 480]
    assert all(chunk["actor_input"] == planner.collect.build_actor_input(
        chunk["queries"], len(chunk["queries"]) * 40) for chunk in chunks)


def test_greedy_assignment():
    chunks = [{"safe_bound": Decimal(amount)} for amount in ("0.03", "0.07", "0.04", "0.02")]
    remaining = {1: Decimal("0.11"), 3: Decimal("0.06")}
    reserve = Decimal("0.02")
    committed = planner.assign_chunks(chunks, remaining, reserve)
    assert [chunk["token_index"] for chunk in chunks] == [None, 1, 3, 1]
    assert committed == {1: Decimal("0.09"), 3: Decimal("0.04")}
    assert all(committed[index] + reserve <= amount for index, amount in remaining.items())
    assert planner.assign_chunks(chunks, {4: Decimal("0.01")}, reserve) == {4: Decimal(0)}
    assert all(chunk["token_index"] is None for chunk in chunks)


def test_auto_search():
    rows = fake_rows()
    remaining = {1: Decimal("0.045"), 3: Decimal("0.025")}
    days, _, chunks, _ = planner.choose_plan(rows, 40, remaining, Decimal(0), "3")
    assert days == 3 and sum(chunk["token_index"] is None for chunk in chunks) == 2
    days, cells, chunks, committed = planner.choose_plan(rows, 40, remaining, Decimal(0))
    assert days == 2 and all(chunk["token_index"] is not None for chunk in chunks)
    assert all(len(cell["queries"]) == 2 for cell in cells)
    assert committed == {1: Decimal("0.044"), 3: Decimal("0.022")}
    days, _, chunks, _ = planner.choose_plan(rows, 40, {1: Decimal(1)}, Decimal(0))
    assert days == 14 and all(chunk["token_index"] == 1 for chunk in chunks)
    days, cells, chunks, committed = planner.choose_plan(rows, 40, {1: Decimal("0.01")}, Decimal(0))
    assert days == 1 and all(chunk["token_index"] is None for chunk in chunks)
    output = io.StringIO()
    with redirect_stdout(output):
        planner.print_plan(planner.keyword_order(rows), cells, chunks, committed,
                           {1: Decimal("0.01")}, Decimal(0), days, 40)
    assert "Shortfall: 3 unassigned chunks ($0.033000 safe bound)" in output.getvalue()
    assert "$0.001000 more free on one token" in output.getvalue()
    days, cells, chunks, _ = planner.choose_plan([], 40, remaining, Decimal(0))
    assert days == 14 and cells == chunks == []
    days, cells, _, _ = planner.choose_plan(rows, 20, {1: Decimal(10)}, Decimal(0), "40")
    assert days == 40 and all(cell["sampled_days"] == list(range(1, 32)) for cell in cells)


def test_offline_remaining():
    with patch.dict(sys.modules, {"apify_client": None}), \
            patch.object(planner.collect, "parse_tokens", unexpected_call):
        assert planner.read_remaining([1, 3, 4], "3.95,2.13,1.62") == {
            1: Decimal("3.95"), 3: Decimal("2.13"), 4: Decimal("1.62")
        }
    for raw in ("1,2", "NaN", "Infinity", "-1"):
        try:
            planner.read_remaining([1], raw)
        except ValueError:
            pass
        else:
            assert False, "Invalid credit must be rejected"


def test_read_only_credit_models():
    calls = []
    tokens = ["private-alpha", "private-beta", "private-gamma"]

    def client(token):
        calls.append(token)
        return SimpleNamespace(user=lambda: SimpleNamespace(limits=lambda: SimpleNamespace(
            limits=SimpleNamespace(max_monthly_usage_usd=5),
            current=SimpleNamespace(monthly_usage_usd=1.05))))

    with patch.dict(sys.modules, {"apify_client": SimpleNamespace(ApifyClient=client)}), \
            patch.dict(planner.os.environ, APIFY_TOKEN=",".join(tokens)):
        assert planner.read_remaining([3, 1], None) == {3: Decimal("3.95"), 1: Decimal("3.95")}
    assert calls == [tokens[2], tokens[0]]


def test_main_outputs_without_credentials_or_network():
    csv_stream = io.StringIO()
    output = io.StringIO()
    log = "keyword,month,capped\nmbg,2025-01,True\nsppg,2025-01,True\nbgn,2025-01,True\n"
    tokens = ["private-alpha", "private-beta", "private-gamma"]
    with patch.object(Path, "open", autospec=True) as open_file, \
            patch.object(Path, "write_text", autospec=True) as write_text, \
            patch.dict(sys.modules, {"apify_client": None}), \
            patch.dict(planner.os.environ, APIFY_TOKEN=",".join(tokens)), \
            patch.object(planner.collect, "parse_tokens", unexpected_call), \
            redirect_stdout(output):
        open_file.return_value.__enter__.side_effect = [io.StringIO(log), csv_stream]
        planner.main(["--tokens", "1,3", "--remaining", "0.045,0.025", "--reserve", "0"])
    assert [str(call.args[0]) for call in open_file.call_args_list] == [
        "data/collection_log.csv", "data/extra_pull_plan.csv"]
    assert str(write_text.call_args.args[0]) == "data/extra_pull_plan.json"
    raw_json = write_text.call_args.args[1]
    saved = json.loads(raw_json)
    assert saved["parameters"] == {
        "days_per_cell": 2, "items_per_day": 40, "reserve": 0, "price": 0.00025,
        "overshoot": 1.1, "tokens": [1, 3], "remaining": {"1": 0.045, "3": 0.025},
    }
    assert saved["notes"] == [
        "Each day-window returns the newest items of that UTC day, so the last hours of the UTC day (Indonesian night) are over-represented; sub-day bounds do not work.",
        "New posts must carry a pull_batch marker (day_sample) when merged; the earlier month-wide pull is popularity-skewed.",
    ]
    assert len(saved["chunks"]) == 3
    assert all(set(chunk) == {"keyword", "months", "queries", "actor_input", "token_index",
                              "items_cap", "cost_bound", "safe_bound"}
               for chunk in saved["chunks"])
    assert "APIFY_TOKEN" not in raw_json
    assert all(token not in raw_json + output.getvalue() for token in tokens)
    assert csv_stream.getvalue().splitlines()[0] == (
        "keyword,month,sampled_days,queries,items_per_day,items_cap,cost_bound_usd"
    )
    rows = list(csv.DictReader(io.StringIO(csv_stream.getvalue())))
    assert [row["keyword"] for row in rows] == ["mbg", "sppg", "bgn"]
    assert all(row["sampled_days"] == "8;24" and row["queries"] == "2"
               and row["items_per_day"] == "40" and row["items_cap"] == "80"
               and row["cost_bound_usd"] == "0.020000" for row in rows)
    assert "TOTAL" in output.getvalue() and "Committed safe USD" in output.getvalue()
    assert "Chosen D: 2" in output.getvalue()
    assert output.getvalue().endswith("Dry run: nothing fetched, nothing spent.\n")


if __name__ == "__main__":
    with patch.object(planner.collect, "fetch", unexpected_call), \
            patch.object(planner.collect.judge, "load_encoder", unexpected_call):
        for name, test in list(globals().items()):
            if name.startswith("test_") and callable(test):
                test()
    print("ok")
