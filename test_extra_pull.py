import csv
import io
import json
import re
import socket
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import extra_pull as planner


def unexpected_call(*args, **kwargs):
    assert False, "No network, model loading, or fetching is allowed"


def fake_rows(keywords=("mbg", "sppg", "bgn")):
    return [{"keyword": keyword, "month": "2025-01", "capped": "True"}
            for keyword in keywords]


def test_sample_slots():
    assert (planner.SLOT_HOURS, planner.ROTATION_STEP, planner.PHASE_STEP) == (6, 5, 7)
    for month, length, phase in (("2025-02", 28, 7), ("2025-04", 30, 21), ("2025-01", 31, 0)):
        start, following = planner.month_bounds(month)
        for count in (1, 12, length):
            slots = planner.sample_slots(month, count)
            days = [(2 * k + 1) * length // (2 * count) for k in range(count)]
            expected = [min(start + timedelta(days=day, hours=(phase + k * 5) % 24),
                            following - timedelta(hours=6)) for k, day in enumerate(days)]
            assert [a for a, _ in slots] == expected
            assert len(slots) == len({a for a, _ in slots}) == count
            assert all(start <= a < b <= following and b - a == timedelta(hours=6)
                       and a.tzinfo == timezone.utc for a, b in slots)
            gaps = [b - a for a, b in zip(days, days[1:])]
            assert not gaps or max(gaps) - min(gaps) <= 1
            if count >= 8:
                assert len({a.hour for a, _ in slots}) >= 8
        for count in (0, -1, length + 1):
            try:
                planner.sample_slots(month, count)
            except ValueError:
                pass
            else:
                assert False, "D must be positive and no larger than L"


def test_rotation_clamping_and_rollover():
    january = planner.sample_slots("2025-01", 12)
    february = planner.sample_slots("2025-02", 12)
    assert [a.hour for a, _ in january] == [(k * 5) % 24 for k in range(12)]
    assert [a.hour for a, _ in february] == [(7 + k * 5) % 24 for k in range(12)]
    assert january[0][0].isoformat() == "2025-01-02T00:00:00+00:00"
    assert february[0][0].isoformat() == "2025-02-02T07:00:00+00:00"
    # February's final unshifted start is 22:00; shift back four hours to keep six hours.
    start, end = planner.sample_slots("2025-02", 28)[-1]
    assert start.isoformat() == "2025-02-28T18:00:00+00:00"
    assert end.isoformat() == "2025-03-01T00:00:00+00:00"
    assert end - start == timedelta(seconds=21600)
    assert planner.month_bounds("2025-12") == (
        datetime(2025, 12, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, tzinfo=timezone.utc))
    december = datetime(2025, 12, 31, 18, tzinfo=timezone.utc)
    assert planner.build_query("mbg", [], december, december + timedelta(hours=6)) == (
        "mbg since_time:1767204000 until_time:1767225600")


def test_query_formatting():
    start = datetime(2025, 1, 31, 22, tzinfo=timezone.utc)
    end = start + timedelta(hours=6)
    assert planner.timegm(start.utctimetuple()) == 1738360800
    assert planner.build_query("bgn", ["mbg", "makan bergizi gratis", "sppg"], start, end) == (
        'bgn -mbg -"makan bergizi gratis" -sppg since_time:1738360800 until_time:1738382400')
    query = planner.build_query("makan bergizi gratis", [], start, end)
    assert query == '"makan bergizi gratis" since_time:1738360800 until_time:1738382400'
    assert " since:" not in query and " until:" not in query


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
    cells = planner.plan_cells(rows, 12)
    assert [(cell["keyword"], cell["month"]) for cell in cells] == [
        ("sppg", "2025-01"), ("mbg", "2025-01"), ("mbg", "2025-02"),
        ("makan bergizi gratis", "2025-01"), ("embege", "2025-01"),
    ]
    assert cells[2]["slot_starts_utc"][0] == "2025-02-02T07:00"
    assert cells[3]["queries"][0].startswith('"makan bergizi gratis" -sppg -mbg since_time:')
    assert cells[4]["queries"][0].startswith('embege -sppg -mbg -"makan bergizi gratis" -bgn since_time:')
    queries = [query for cell in cells for query in cell["queries"]]
    assert len(queries) == len(set(queries)) == 60
    for cell in cells:
        month_start, month_end = planner.month_bounds(cell["month"])
        for query in cell["queries"]:
            assert " since:" not in query and " until:" not in query
            bounds = re.search(r" since_time:([0-9]+) until_time:([0-9]+)$", query)
            assert bounds
            since, until = map(int, bounds.groups())
            assert until - since == 21600
            assert planner.timegm(month_start.utctimetuple()) <= since
            assert until <= planner.timegm(month_end.utctimetuple())


def test_query_self_checks():
    for bad_query in (
        "mbg since:2025-01-01 until:2025-01-02",
        "mbg since_time:1738360800.0 until_time:1738382400",
        "mbg since_time:1738360800 until_time:1738382400.0",
        "mbg since_time:1736985600 until_time:1736989200",
        "mbg since_time:1736985600 until_time:1737072000",
        "mbg since_time:1735686000 until_time:1735707600",
        "mbg since_time:1738360800 until_time:1738382400",
    ):
        with patch.object(planner, "build_query", return_value=bad_query):
            try:
                planner.plan_cells(fake_rows(("mbg",)), 1)
            except AssertionError:
                pass
            else:
                assert False, "Slot self-checks must reject invalid bounds"
    with patch.object(planner, "build_query", return_value=(
            "mbg since_time:1736985600 until_time:1737007200")):
        try:
            planner.plan_cells(fake_rows(("mbg",)), 12)
        except AssertionError:
            pass
        else:
            assert False, "Duplicate slot starts must be rejected"
    try:
        planner.plan_cells(fake_rows(("mbg", "mbg")), 1)
    except AssertionError:
        pass
    else:
        assert False, "Duplicate queries must be rejected"


def test_items_per_slot():
    cells = planner.plan_cells(fake_rows(), 12)
    assert planner.ITEMS_PER_SLOT == 20
    for cap in (20, 40, 41):
        chunks = planner.make_chunks(cells, cap)
        assert all(chunk["actor_input"]["maxItems"] == cap for chunk in chunks)
        assert all(chunk["items_cap"] == 12 * cap for chunk in chunks)
        assert all(chunk["cost_bound"] == 12 * cap * planner.PRICE for chunk in chunks)
        assert all(chunk["safe_bound"] == 12 * cap * planner.PRICE * planner.OVERSHOOT
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
            planner.main(["--items-per-slot", "19"])
        except SystemExit as error:
            assert error.code == 2
        else:
            assert False, "CLI must reject fewer than 20 items before reading credit"


def test_chunking():
    rows = [{"keyword": "mbg", "month": f"2025-{month:02}", "capped": "True"}
            for month in range(8, 0, -1)]
    cells = planner.plan_cells(rows, 12)
    chunks = planner.make_chunks(cells)
    assert [len(chunk["months"]) for chunk in chunks] == [6, 2]
    assert chunks[0]["months"] == [f"2025-{month:02}" for month in range(1, 7)]
    assert chunks[1]["months"] == ["2025-07", "2025-08"]
    assert [len(chunk["queries"]) for chunk in chunks] == [72, 24]
    assert [chunk["items_cap"] for chunk in chunks] == [1440, 480]
    assert all(chunk["actor_input"] == planner.collect.build_actor_input(
        chunk["queries"], len(chunk["queries"]) * 20) for chunk in chunks)


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
    slots, _, chunks, _ = planner.choose_plan(rows, 20, remaining, Decimal(0), "5")
    assert slots == 5 and sum(chunk["token_index"] is None for chunk in chunks) == 2
    slots, cells, chunks, committed = planner.choose_plan(rows, 20, remaining, Decimal(0), "auto")
    assert slots == 4 and all(chunk["token_index"] is not None for chunk in chunks)
    assert all(len(cell["queries"]) == 4 for cell in cells)
    assert committed == {1: Decimal("0.044"), 3: Decimal("0.022")}
    slots, _, chunks, _ = planner.choose_plan(rows, 20, {1: Decimal(1)}, Decimal(0), "auto")
    assert slots == 31 and all(chunk["token_index"] == 1 for chunk in chunks)
    rows.append({"keyword": "mbg", "month": "2025-02", "capped": "True"})
    slots, _, _, _ = planner.choose_plan(rows, 20, {1: Decimal(10)}, Decimal(0), "auto")
    assert slots == 28
    try:
        planner.choose_plan(rows, 20, remaining, Decimal(0), "29")
    except ValueError:
        pass
    else:
        assert False, "D cannot exceed the shortest capped month"
    rows = fake_rows()
    slots, cells, chunks, committed = planner.choose_plan(rows, 20, {1: Decimal("0.005")}, Decimal(0), "auto")
    assert slots == 1 and all(chunk["token_index"] is None for chunk in chunks)
    output = io.StringIO()
    with redirect_stdout(output):
        planner.print_plan(planner.keyword_order(rows), cells, chunks, committed,
                           {1: Decimal("0.005")}, Decimal(0), slots, 20)
    assert "Shortfall: 3 unassigned chunks ($0.016500 safe bound)" in output.getvalue()
    assert "$0.000500 more free on one token" in output.getvalue()


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
    for arguments, count in (([], 12), (["--slots-per-cell", "auto"], 4)):
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
            planner.main(["--tokens", "1,3", "--remaining", "0.045,0.025", "--reserve", "0"] + arguments)
        assert [str(call.args[0]) for call in open_file.call_args_list] == [
            "data/collection_log.csv", "data/extra_pull_plan.csv"]
        assert str(write_text.call_args.args[0]) == "data/extra_pull_plan.json"
        raw_json = write_text.call_args.args[1]
        saved = json.loads(raw_json)
        assert saved["parameters"] == {
            "slots_per_cell": count, "items_per_slot": 20, "slot_hours": 6,
            "rotation_step": 5, "phase_step": 7, "reserve": 0, "price": 0.00025,
            "overshoot": 1.1, "tokens": [1, 3], "remaining": {"1": 0.045, "3": 0.025},
        }
        assert saved["notes"] == [
            "Each slot returns the newest items inside its 6-hour UTC window; busy slots return a cluster of consecutive posts. Slot start hours rotate so every hour of the day is covered.",
            "The actor cannot filter language: non-Indonesian items are paid for and dropped.",
            "New posts must carry a pull_batch marker (slot_sample) when merged; the earlier month-wide pull is popularity-skewed.",
        ]
        assert len(saved["chunks"]) == 3
        assert all(set(chunk) == {"keyword", "months", "queries", "actor_input", "token_index",
                                  "items_cap", "cost_bound", "safe_bound"}
                   for chunk in saved["chunks"])
        assert "APIFY_TOKEN" not in raw_json
        assert all(token not in raw_json + csv_stream.getvalue() + output.getvalue() for token in tokens)
        assert csv_stream.getvalue().splitlines()[0] == (
            "keyword,month,slots,slot_starts_utc,items_per_slot,items_cap,cost_bound_usd")
        rows = list(csv.DictReader(io.StringIO(csv_stream.getvalue())))
        assert [row["keyword"] for row in rows] == ["mbg", "sppg", "bgn"]
        expected_starts = ";".join(start.strftime("%Y-%m-%dT%H:%M")
                                   for start, _ in planner.sample_slots("2025-01", count))
        assert all(row["slot_starts_utc"] == expected_starts and row["slots"] == str(count)
                   and row["items_per_slot"] == "20" and row["items_cap"] == str(count * 20)
                   and row["cost_bound_usd"] == f"{count * 20 * planner.PRICE:.6f}" for row in rows)
        assert "TOTAL" in output.getvalue() and "Committed safe USD" in output.getvalue()
        assert f"Chosen slots per cell: {count}\n" in output.getvalue()
        assert output.getvalue().endswith("Dry run: nothing fetched, nothing spent.\n")


if __name__ == "__main__":
    with patch.object(planner.collect, "fetch", unexpected_call), \
            patch.object(socket.socket, "connect", unexpected_call), \
            patch.object(socket, "create_connection", unexpected_call):
        for name, test in list(globals().items()):
            if name.startswith("test_") and callable(test):
                test()
    print("ok")
