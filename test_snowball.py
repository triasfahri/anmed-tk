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

import numpy as np

import collect
import validate
from judge import (
    auc,
    calibrate,
    load_calibration,
    mask_terms,
    prep,
    save_calibration,
    score,
    youden_threshold,
)


def fake_encode(texts):
    vectors = np.zeros((len(texts), 64), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in text.split():
            index = int.from_bytes(hashlib.md5(token.encode()).digest(), "big") % 64
            vectors[i, index] += 1
    return vectors


def calibration_texts():
    R = [
        "MBG siswa sekolah menerima makanan bergizi dari dapur sehat",
        "MBG dapur sekolah memasak nasi sayur untuk siswa",
        "MBG makanan sekolah harus bergizi sehat dan bersih",
        "MBG siswa keracunan makanan dari dapur sekolah",
        "MBG dapur sehat menyiapkan makanan bergizi untuk sekolah",
        "MBG guru mengawasi makanan siswa di sekolah",
        "MBG sekolah memeriksa kebersihan dapur dan makanan",
        "MBG menu makanan bergizi berisi nasi telur sayur",
        "MBG siswa mendapat makanan sehat sebelum belajar sekolah",
        "MBG keracunan membuat dapur sekolah berhenti memasak makanan",
        "MBG orangtua meminta makanan sekolah sehat dan bergizi",
        "MBG dapur mengirim makanan bergizi ke sekolah siswa",
    ]
    B = [
        "Harga bensin naik mengikuti pasar minyak dunia",
        "Kemacetan panjang memenuhi jalan tol menuju kota",
        "Klub sepakbola menang telak dalam pertandingan liga",
        "Hujan deras menyebabkan banjir di kawasan perumahan",
        "Konser musik menghadirkan penyanyi terkenal malam minggu",
        "Ponsel terbaru menawarkan kamera tajam serta baterai besar",
        "Wisata pantai ramai dikunjungi turis saat liburan",
        "Bursa saham bergerak turun karena sentimen ekonomi",
        "Film petualangan tayang perdana di bioskop besok",
        "Kereta terlambat tiba akibat perbaikan rel stasiun",
        "Tim bulutangkis meraih medali emas kejuaraan internasional",
        "Sepeda motor listrik mendapat potongan pajak kendaraan",
    ]
    return R, B


def test_mask_terms():
    terms = ["mbg", "makan bergizi gratis"]
    text = "MBG #MBG mbg MaKaN BERGIZI GRATIS #MakanBergiziGratis mbgx ambg"
    assert mask_terms(text, terms).split() == ["mbgx", "ambg"]
    assert mask_terms("makan\tbergizi\nGRATIS", terms).strip() == ""
    assert mask_terms("#MakanBergiziGratisx xmakan bergizi gratis", terms).split() == [
        "#MakanBergiziGratisx", "xmakan", "bergizi", "gratis"
    ]
    assert mask_terms("makan bergizi gratis", ["makan", "makan bergizi gratis"]) == ""
    assert mask_terms("a+b A+B aab", ["a+b"]).split() == ["aab"]


def test_prep():
    text = "  MBG #SEKOLAH @pengguna HTTPS://contoh.id/berita  MAKAN\nSehat  "
    assert prep(text, ["mbg"]) == "sekolah @pengguna makan sehat"
    assert prep("#MakanBergiziGratis #Sehat", ["makan bergizi gratis"]) == "sehat"


def test_calibrate_separates():
    R, B = calibration_texts()
    mask = ["mbg", "makan bergizi gratis"]
    cal = calibrate(R, B, fake_encode, mask)
    R_scores = score(R, cal, fake_encode, mask)
    B_scores = score(B, cal, fake_encode, mask)
    assert set(cal) == {"mu", "c", "tau", "auc", "b0"}
    assert cal["auc"] >= 0.95
    assert cal["b0"] <= 0.2
    assert np.median(B_scores) <= cal["tau"] <= np.median(R_scores)
    assert cal["auc"] == auc(R_scores, B_scores)
    assert cal["b0"] == np.mean(B_scores >= cal["tau"])
    assert np.allclose(np.linalg.norm(cal["c"]), 1)
    prepared = [prep(text, mask) for text in R + B]
    assert np.allclose(cal["mu"], fake_encode(prepared).mean(axis=0))
    with_short = calibrate(R + ["MBG"], B + ["dua kata"], fake_encode, mask)
    for key in cal:
        assert np.array_equal(cal[key], with_short[key])


def test_score_short_text_nan():
    R, B = calibration_texts()
    cal = calibrate(R, B, fake_encode, ["mbg"])
    embedded = []

    def encode(texts):
        embedded.extend(texts)
        return fake_encode(texts)

    texts = ["MBG #MBG", "dua kata", R[0], ""]
    scores = score(texts, cal, encode, ["mbg"])
    assert scores.shape == (len(texts),)
    assert np.isnan(scores[[0, 1, 3]]).all()
    assert np.isfinite(scores[2])
    assert not (scores[[0, 1, 3]] >= cal["tau"]).any()
    assert embedded == [prep(R[0], ["mbg"])]
    embedded.clear()
    assert np.isnan(score(["MBG"], cal, encode, ["mbg"])).all()
    assert score([], cal, encode).shape == (0,)
    assert embedded == []


def test_score_zero_norms():
    texts = ["siswa makan sehat"]
    embedding = fake_encode(texts)[0]
    cal = {"mu": embedding, "c": np.ones(64, dtype=np.float32)}
    assert score(texts, cal, fake_encode).tolist() == [0.0]
    cal = {"mu": np.zeros(64, dtype=np.float32), "c": np.zeros(64, dtype=np.float32)}
    assert score(texts, cal, fake_encode).tolist() == [0.0]


def test_youden_and_auc():
    pos, neg = np.array([0.7, 0.8, 0.9]), np.array([0.1, 0.2, 0.3])
    assert auc(pos, neg) == 1.0
    assert neg.max() < youden_threshold(pos, neg) <= pos.min()
    assert auc(pos, pos) == 0.5
    assert auc([1, 2], [0, 1]) == 0.875
    assert auc(neg, pos) == 0.0
    assert youden_threshold([1, 3], [0, 2]) == 3
    assert youden_threshold(pos, pos) == pos.max()
    assert youden_threshold([0, 0, 1], [0, 0, 1]) == 1


def test_calibration_roundtrip():
    R, B = calibration_texts()
    cal = calibrate(R, B, fake_encode, ["mbg"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "judge.npz"
        save_calibration(cal, path)
        loaded = load_calibration(path)
        assert list(Path(directory).iterdir()) == [path]
    assert loaded.keys() == cal.keys()
    for key in ("mu", "c"):
        assert np.array_equal(loaded[key], cal[key])
    for key in ("tau", "auc", "b0"):
        assert isinstance(loaded[key], float)
        assert loaded[key] == cal[key]


def test_calibrate_raises_when_not_separable():
    R, _ = calibration_texts()
    try:
        calibrate(R, R, fake_encode, ["mbg", "makan bergizi gratis"])
    except ValueError as error:
        assert str(error).startswith("judge does not separate")
        assert "switch ENCODER" in str(error)
    else:
        assert False, "identical calibration groups should fail"


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


def test_extract_candidates():
    rows = [{"text": "MBG #MBG dapur sehat dapur sehat #Gizi #Gizi dan 123 xy @pengguna "
                     "https://example.com/janganambil"} for _ in range(6)]
    rows += [{"text": "dapur sehat langka"}, {"text": "sekolah dan bersih"}]
    candidates = {term: df for term, df, lift in collect.extract_candidates(rows, collect.SEEDS, 100, 5)}
    assert candidates["dapur sehat"] == 7
    assert candidates["#gizi"] == 6
    assert candidates["dapur"] == 7
    for term in ("mbg", "#mbg", "dan", "123", "xy", "pengguna", "langka", "janganambil"):
        assert term not in candidates
    assert "sekolah bersih" not in {term for term, df, lift in collect.extract_candidates(rows, [], 100, 1)}
    assert len(collect.extract_candidates(rows, collect.SEEDS, 2, 5)) == 2
    assert collect.extract_candidates(rows, collect.SEEDS, 100, 9) == []
    boundary = {term for term, df, lift in collect.extract_candidates(
        [{"text": "#MBG mbgx"}], ["MBG"], 20, 1
    )}
    assert "#mbg" not in boundary and "mbgx" in boundary


def test_candidates_exclude_subphrases():
    rows = [{"text": "Makan bergizi gratis bergizi gratis makan bergizi #Bergizi sehat"}]
    terms = {term for term, df, lift in collect.extract_candidates(rows, ["makan bergizi gratis"], 100, 1)}
    for term in ("bergizi gratis", "makan bergizi", "makan", "bergizi", "gratis", "#bergizi"):
        assert term not in terms
    assert "sehat" in terms


def test_extract_candidates_lift():
    rows = [{"text": "program dan bgn dan bgn dan dapur dan sppg" +
             (" dan alias" if i < 3 else "")} for i in range(4)]
    rows += [{"text": "program"}] * 2
    background = [{"text": "program" + (" dan sppg" if i == 0 else "")} for i in range(6)]
    candidates = collect.extract_candidates(rows, [], 25, 3, background=background)
    assert all(isinstance(candidate, tuple) and len(candidate) == 3 for candidate in candidates)
    assert [term for term, df, lift in candidates] == ["bgn", "dapur", "alias"]
    assert candidates[0] == ("bgn", 4, (4 / 6) / (1 / 7))
    assert "program" not in {term for term, df, lift in candidates}
    assert "sppg" not in {term for term, df, lift in candidates}
    assert collect.extract_candidates(rows, [], 1, 3, background=background) == candidates[:1]
    assert collect.extract_candidates(rows, [], 25, 5, background=background) == []
    assert collect.extract_candidates([], [], 25, 1, background=background) == []
    unfiltered = collect.extract_candidates(rows, [], 25, 3, background=background, min_lift=0)
    assert {term for term, df, lift in unfiltered} == {"program", "bgn", "dapur", "sppg", "alias"}
    old_order = collect.extract_candidates(rows, [], 25, 3, background=None)
    assert old_order == [("program", 6, None), ("bgn", 4, None), ("dapur", 4, None),
                         ("sppg", 4, None), ("alias", 3, None)]
    known = [{"text": "#MBG dan makan bergizi gratis dan bgn dan bgn"}]
    assert collect.extract_candidates(known, collect.SEEDS, 25, 1, background=[{"text": "program"}] * 4) == [
        ("bgn", 1, 5.0)
    ]


def test_prescreen_and_pilot_examples():
    assert collect.prescreen(collect.MIN_NOVEL_N - 1, 1.0) == "reject_redundant"
    assert collect.prescreen(collect.MIN_NOVEL_N - 1, 0.0) == "reject_redundant"
    assert collect.prescreen(collect.MIN_NOVEL_N, collect.REVIEW_MIN_PRECISION - 0.01) == "reject_precision"
    assert collect.prescreen(collect.MIN_NOVEL_N, collect.REVIEW_MIN_PRECISION) == "review"
    assert collect.prescreen(100, 0.6) == "review"
    novel = [f"sppg @pengguna {i} " + "menu sehat " * 40 for i in range(8)]
    rows = [{"text": text} for text in novel + ["MBG known", "#MakanBergiziGratis known"]]
    random_state = random.getstate()
    examples = collect.pilot_examples(rows, collect.SEEDS, k=3, seed=7)
    assert random.getstate() == random_state
    assert examples == [text[:240] for text in random.Random(7).sample(novel, 3)]
    assert examples == collect.pilot_examples(rows, collect.SEEDS, k=3, seed=7)
    assert len(examples) == 3 and all(len(text) <= 240 for text in examples)
    assert all(mask_terms(text, collect.SEEDS) == text for text in examples)
    assert len(collect.pilot_examples(rows[:2], collect.SEEDS, k=3)) == 2
    assert collect.pilot_examples(rows, collect.SEEDS, k=0) == []
    assert collect.pilot_examples(rows[-2:], collect.SEEDS) == []
    assert collect.pilot_examples([], collect.SEEDS) == []


def test_approve_terms():
    shortlist = [{"round": 1, "term": term, "df": 40, "lift": 12.3, "novel_n": 84,
                  "precision_novel": precision, "examples": [f"{term} snippet {i}" for i in range(3)]}
                 for term, precision in [("sppg", 0.88), ("bgn", 0.8), ("siswa", 0.6)]]
    for lines, expected, invalid in [
        (["1,3"], (["sppg", "siswa"], []), 0),
        (["2;dapur umum, bgn, dapur umum, ,"], (["bgn"], ["dapur umum", "bgn"]), 0),
        (["bad", "0", "4", "1," , "2"], (["bgn"], []), 4),
        ([""], ([], []), 0),
        ([";dapur umum"], ([], ["dapur umum"]), 0),
    ]:
        output = []
        answers = iter(lines)
        assert collect.approve_terms(shortlist[::-1], answers.__next__, output.append) == expected
        assert output[0] == ("Round 1 review: 3 candidate terms. "
                             "Would a post containing ONLY this term be about MBG?")
        assert output[1] == "  1. sppg   df=40 lift=12.3 novel=84 judge-pass=0.88"
        assert all(f'       - "{row["term"]} snippet {i}"' in output
                   for row in shortlist for i in range(3))
        assert sum(line.startswith("Invalid selection") for line in output) == invalid
        assert sum(line.startswith("Accept (") for line in output) == len(lines)
    assert collect.approve_terms([], lambda: ";bgn", lambda line: None) == ([], ["bgn"])


def test_approve_terms_title():
    output = []
    assert collect.approve_terms([], lambda: ";bgn", output.append, title="Final pick") == ([], ["bgn"])
    assert output[0] == ("Final pick: 0 candidate terms. "
                         "Would a post containing ONLY this term be about MBG?")
    assert all("Round" not in line for line in output)
    output = []
    collect.approve_terms([], lambda: "", output.append)
    assert output[0].startswith("Review: 0 candidate terms.")


def test_pilot_stats_and_decisions():
    R, B = calibration_texts()
    cal = calibrate(R, B, fake_encode, collect.SEEDS)
    positive = {"text": "sppg " + mask_terms(R[0], collect.SEEDS)}
    negative = {"text": "sppg " + B[0]}
    known = [{"text": "#MBG " + B[1]}, {"text": "#MakanBergiziGratis " + B[2]}]
    novel_n, precision = collect.pilot_stats(
        [positive] * 16 + [negative] * 4 + known, collect.SEEDS, "sppg", cal, fake_encode
    )
    assert (novel_n, precision) == (20, 0.8)
    assert collect.decide(novel_n, precision) == "accept"
    assert collect.decide(19, 1.0) == "reject_redundant"
    assert collect.decide(20, 0.6) == "reject_precision"
    assert collect.decide(20, 0.7) == "accept"
    assert collect.pilot_stats([positive] * 12 + [negative] * 8,
                               collect.SEEDS, "sppg", cal, fake_encode) == (20, 0.6)
    assert collect.pilot_stats([positive] * 16 + [{"text": "sppg"}] * 4,
                               collect.SEEDS, "sppg", cal, fake_encode) == (20, 0.8)
    assert collect.pilot_stats(known, collect.SEEDS, "sppg", cal, fake_encode) == (0, 0.0)
    results = [
        {"term": str(i), "decision": "accept", "precision_novel": 0.8, "novel_n": 20 + i}
        for i in range(7)
    ]
    results += [{"term": "best", "decision": "accept", "precision_novel": 0.9, "novel_n": 20},
                {"term": "rejected", "decision": "reject_precision",
                 "precision_novel": 1.0, "novel_n": 100}]
    assert collect.pick_accepted(results, 5) == ["best", "6", "5", "4", "3"]
    assert collect.pick_accepted(results, 0) == []


def test_gain_stop_and_cost():
    assert collect.gain(10, 20) == 0.5
    assert collect.gain(0, 0) == float("inf")
    assert collect.should_stop(1, 0.0) is None
    assert collect.should_stop(2, 0.01) == "low_gain"
    assert collect.should_stop(2, 0.5) is None
    assert collect.should_stop(2, collect.STOP_GAIN) is None
    assert collect.should_stop(4, 0.5) == "max_rounds"
    assert collect.should_stop(1, 0.5, 0) == "no_accepted"
    assert collect.cost(1000) == collect.PRICE_PER_1000 == 0.25
    assert collect.cost(0) == 0.0


def test_dry_run_and_missing_secrets():
    script = str(Path(__file__).with_name("collect.py"))
    env = dict(os.environ, APIFY_TOKEN="dummy-token", USER_HASH_SALT="dummy-salt")
    result = subprocess.run([sys.executable, script, "--dry-run"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Tokens loaded: 1" in result.stdout
    queries = [q for term in collect.SEEDS for q in collect.build_queries(
        term, collect.month_windows(collect.START, collect.END)
    )]
    assert "Round 1 queries: 44 (2 seeds x 22 months)" in result.stdout
    assert queries[0] in result.stdout and queries[-1] in result.stdout
    discovery_n = len(queries) * collect.MAX_PER_QUERY_DISCOVERY
    assert f"44 x {collect.MAX_PER_QUERY_DISCOVERY} = {discovery_n} items" in result.stdout
    assert f"${collect.cost(discovery_n):.6f}" in result.stdout
    assert f"Background ({collect.BACKGROUND_N} items)" in result.stdout
    assert collect.build_queries(collect.BACKGROUND_QUERY, [(collect.START, collect.END)])[0] in result.stdout
    bound = discovery_n + collect.BACKGROUND_N
    assert f"Round 1 collection cost bound: ${collect.cost(bound):.6f}" in result.stdout
    assert (f"Including up to {collect.TOP_N_CANDIDATES} pilots: "
            f"${collect.cost(bound + collect.TOP_N_CANDIDATES * collect.PILOT_N):.6f}") in result.stdout
    assert "dummy-token" not in result.stdout + result.stderr
    assert "dummy-salt" not in result.stdout + result.stderr
    env.pop("APIFY_TOKEN")
    env.pop("USER_HASH_SALT")
    result = subprocess.run([sys.executable, script, "--dry-run"], env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "APIFY_TOKEN" in result.stderr and "USER_HASH_SALT" in result.stderr
    assert "sentence_transformers" not in result.stderr


def test_dry_run_token_list_and_placeholder():
    for raw, count in (("a1,b2,c3", 3), ("PASTE_YOUR_TOKEN_HERE", None)):
        output, errors = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, APIFY_TOKEN=raw, USER_HASH_SALT="private-salt"), \
                patch.object(collect, "rotating_fetch", unexpected_call), \
                patch.object(collect.judge, "load_encoder", unexpected_call), \
                redirect_stdout(output), redirect_stderr(errors):
            try:
                collect.main(["--dry-run"])
            except SystemExit as error:
                assert count is None and error.code != 0
                assert raw not in str(error)
            else:
                assert count is not None
                assert f"Tokens loaded: {count}" in output.getvalue()
        assert all(token not in output.getvalue() + errors.getvalue() for token in raw.split(","))
        assert "private-salt" not in output.getvalue() + errors.getvalue()
        if count is None:
            assert "placeholder" in errors.getvalue()


def test_discovery_cli_review_and_auto():
    with patch.dict(os.environ, APIFY_TOKEN="private-token", USER_HASH_SALT="private-salt"), \
            patch.object(collect.sys.stdin, "isatty", return_value=False), \
            patch.object(collect, "rotating_fetch", unexpected_call), \
            patch.object(collect.judge, "load_encoder", unexpected_call), \
            redirect_stderr(io.StringIO()) as errors:
        try:
            collect.main(["--yes"])
        except SystemExit as error:
            assert error.code == 2
            assert "human review needs a terminal; run it in a terminal tab or pass --auto" in errors.getvalue()
        else:
            assert False, "--yes must still require a terminal for human review"
    with patch.dict(os.environ, APIFY_TOKEN="private-token", USER_HASH_SALT="private-salt"), \
            patch.object(collect, "rotating_fetch", return_value=unexpected_call), \
            patch.object(collect.judge, "load_encoder", unexpected_call), \
            patch.object(collect, "discover") as discover_fn, \
            patch.object(collect, "final") as final_fn:
        with patch.object(collect.sys.stdin, "isatty", return_value=False):
            collect.main(["--auto", "--yes"])
            assert discover_fn.call_args.args[:2] == (Path("data"), unexpected_call)
            assert discover_fn.call_args.kwargs == {"review": None}
            collect.main(["--final", "--yes"])
            final_fn.assert_called_once()
        discover_fn.reset_mock()
        with patch.object(collect.sys.stdin, "isatty", return_value=True):
            collect.main(["--yes"])
        assert discover_fn.call_args.kwargs == {"review": collect.approve_terms}


def test_parse_args():
    with patch.dict(os.environ, APIFY_TOKEN="a1,b2", USER_HASH_SALT="y"), \
            patch.object(collect.sys.stdin, "isatty", return_value=False):
        args = collect.parse_args(["--overnight"])
        assert args.auto and args.yes and args.max_rounds == 3
        assert args.tokens == ["a1", "b2"] and args.salt == "y"
        assert collect.parse_args(["--overnight", "--max-rounds", "2"]).max_rounds == 2
        assert collect.parse_args(["--auto"]).max_rounds == collect.MAX_ROUNDS
        for flags in (["--pick"], ["--auto", "--max-rounds", "0"],
                      ["--auto", "--max-rounds", "bad"]):
            with redirect_stderr(io.StringIO()) as errors:
                try:
                    collect.parse_args(flags)
                except SystemExit as error:
                    assert error.code == 2
                else:
                    assert False, "invalid flags or a nonterminal pick must fail"
            if flags == ["--pick"]:
                assert "--pick needs a terminal" in errors.getvalue()
    with patch.dict(os.environ, {}, clear=True), \
            patch.object(collect.sys.stdin, "isatty", return_value=True):
        args = collect.parse_args(["--pick"])
        assert args.pick and args.tokens == [] and args.salt is None
        for mode in ("--final", "--smoke", "--dry-run"):
            with redirect_stderr(io.StringIO()):
                try:
                    collect.parse_args(["--pick", mode])
                except SystemExit as error:
                    assert error.code == 2
                else:
                    assert False, "pick must be exclusive with collection modes"


def test_cli_overnight_and_final_options():
    rounds = []

    def discover(data_dir, fetch_fn, encode, salt, confirm, review):
        assert data_dir == Path("data") and fetch_fn is unexpected_call and salt == "y"
        assert review is None and confirm("cost bound") is True
        rounds.append(collect.MAX_ROUNDS)
        assert collect.should_stop(collect.MAX_ROUNDS, 1.0) == "max_rounds"

    with patch.dict(os.environ, APIFY_TOKEN="a1,b2", USER_HASH_SALT="y"), \
            patch.object(collect.sys.stdin, "isatty", return_value=False), \
            patch.object(collect, "MAX_ROUNDS", collect.MAX_ROUNDS), \
            patch.object(collect, "rotating_fetch", return_value=unexpected_call), \
            patch.object(collect.judge, "load_encoder", unexpected_call), \
            patch("builtins.input", unexpected_call), \
            patch.object(collect, "discover", discover), \
            patch.object(collect, "final") as final_fn:
        collect.main(["--overnight"])
        collect.main(["--overnight", "--max-rounds", "2"])
        collect.main(["--final", "--yes", "--allow-auto"])
        assert final_fn.call_args.kwargs == {"allow_auto": True, "predefined": False}
        collect.main(["--final", "--yes", "--predefined"])
        assert final_fn.call_args.kwargs == {"allow_auto": False, "predefined": True}
    assert rounds == [3, 2]


def test_read_jsonl_survives_unicode_line_separator():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.jsonl"
        collect._append_jsonl(path, [{"id": "1", "text": "a\u2028b"}, {"id": "2", "text": "c"}])
        assert [row["id"] for row in collect._read_jsonl(path)] == ["1", "2"]


def test_predefined_keywords():
    assert collect.KEYWORDS == ["mbg", "makan bergizi gratis", "sppg", "satuan pelayanan pemenuhan gizi",
                                "bgn", "badan gizi nasional", "embege"]
    with tempfile.TemporaryDirectory() as directory:
        run_discovery(directory)
        original = collect.START, collect.END
        collect.START, collect.END = "2025-01-01", "2025-01-31"
        terms = []

        def fetch_fn(queries, max_items):
            terms.append(queries[0].split(" since:")[0])
            return []

        try:
            with redirect_stdout(io.StringIO()):
                collect.final(directory, fetch_fn, fake_encode, "salt", lambda message: True,
                              predefined=True)
        finally:
            collect.START, collect.END = original
        frozen = json.loads((Path(directory) / "keywords_final.json").read_text())
        assert frozen["keywords"] == collect.KEYWORDS and frozen["picked_by"] == "human"
        assert terms == ["mbg", '"makan bergizi gratis"', "sppg", '"satuan pelayanan pemenuhan gizi"', "bgn",
                         '"badan gizi nasional"', "embege"]
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("collect.py")), "--predefined"],
                            env=dict(os.environ, APIFY_TOKEN="a", USER_HASH_SALT="b"),
                            capture_output=True, text=True)
    assert result.returncode != 0 and "--final" in result.stderr


def test_pick_cli_without_secrets_or_encoder():
    with patch.dict(os.environ, {}, clear=True), \
            patch.object(collect.sys.stdin, "isatty", return_value=True), \
            patch.object(collect, "parse_tokens", unexpected_call), \
            patch.object(collect, "rotating_fetch", unexpected_call), \
            patch.object(collect.judge, "load_encoder", unexpected_call), \
            patch.object(collect, "pick") as pick_fn:
        collect.main(["--pick"])
        pick_fn.assert_called_once_with(Path("data"), repick=False)
        pick_fn.reset_mock()
        collect.main(["--pick", "--repick"])
        pick_fn.assert_called_once_with(Path("data"), repick=True)


def discovery_fixture():
    R, B = calibration_texts()
    calls = []
    food = "sppg " + "siswa sekolah makanan dapur bergizi sehat nasi sayur " * 5

    def fetch_fn(queries, max_items):
        calls.append((list(queries), max_items))
        windows = collect.month_windows(collect.START, collect.END)
        if len(queries) == len(collect.SEEDS) * len(windows):
            assert queries == [q for term in collect.SEEDS for q in collect.build_queries(term, windows)]
            assert len(queries) == 44
            assert max_items == collect.MAX_PER_QUERY_DISCOVERY * len(queries)
            assert collect.build_actor_input(queries, max_items)["maxItems"] == 25
            return [raw_tweet(f"seed-{i}", "sppg dan " + R[i % len(R)]
                              + " dan viral @GusDikaOfficial", i % len(queries))
                    for i in range(40)]
        if queries[0].startswith(collect.BACKGROUND_QUERY + " "):
            assert queries == collect.build_queries(collect.BACKGROUND_QUERY, [(collect.START, collect.END)])
            assert max_items == collect.BACKGROUND_N
            return [raw_tweet(f"background-{i}", B[i % len(B)]) for i in range(30)]
        if max_items == collect.PILOT_N:
            assert len(queries) == 1
            if queries[0].startswith("sppg "):
                return [raw_tweet(f"pilot-sppg-{i}", food) for i in range(100)]
            assert queries[0].startswith("viral "), queries
            return [raw_tweet(f"pilot-viral-{i}", "viral " + B[i % len(B)]) for i in range(100)]
        assert queries == collect.build_queries("sppg", windows)
        assert max_items == collect.MAX_PER_QUERY_DISCOVERY * len(queries)
        return [raw_tweet("seed-0", food), raw_tweet("round2-new", food)]

    return fetch_fn, calls


def run_discovery(directory):
    fetch_fn, calls = discovery_fixture()
    original = collect.TOP_N_CANDIDATES
    collect.TOP_N_CANDIDATES = 2
    try:
        with redirect_stdout(io.StringIO()):
            collect.discover(directory, fetch_fn, fake_encode, "test-salt")
    finally:
        collect.TOP_N_CANDIDATES = original
    return calls


def file_snapshot(directory):
    return {path.name: path.read_bytes() for path in Path(directory).iterdir() if path.is_file()}


def unexpected_call(*args):
    assert False, "a completed run must not fetch, encode, or request confirmation"


def test_discover_end_to_end():
    with tempfile.TemporaryDirectory() as directory:
        calls = run_discovery(directory)
        data_dir = Path(directory)
        with (data_dir / "decisions.csv").open() as stream:
            reader = csv.DictReader(stream)
            decisions = list(reader)
            assert reader.fieldnames == ["round", "term", "df", "lift", "novel_n", "precision_novel",
                                         "tau", "b0", "decision", "pilot_cost_usd", "timestamp"]
        by_term = {row["term"]: row for row in decisions}
        assert by_term["sppg"]["decision"] == "accept"
        assert by_term["viral"]["decision"] == "reject_precision"
        assert float(by_term["sppg"]["precision_novel"]) >= collect.MIN_PRECISION_NOVEL
        assert len(decisions) == 2
        assert all(int(row["df"]) == 40 and int(row["novel_n"]) == 100 for row in decisions)
        state = json.loads((data_dir / "state.json").read_text())
        assert state["stop_reason"] == "low_gain"
        assert state["round"] == 2
        assert state["history"][1]["keywords"] == ["sppg"]
        assert state["history"][1]["new_unique"] == 1
        assert state["history"][1]["new_on_topic"] == 1
        frozen = json.loads((data_dir / "keywords_final.json").read_text())
        assert set(frozen) == {"keywords", "encoder", "tau", "params", "frozen_at"}
        assert frozen["keywords"] == collect.SEEDS + ["sppg"]
        assert frozen["tau"] == state["judge"]["tau"]
        assert state["judge"]["auc_r_vs_b"] >= 0.95
        rows = [json.loads(line) for line in (data_dir / "posts.jsonl").read_text().splitlines()]
        assert len(rows) == len({row["id"] for row in rows}) == 41
        ingest_keys = set(collect.minimize(sample_tweet(), "salt"))
        for row in rows:
            assert set(row) == ingest_keys | {"judge_score", "on_topic"}
            assert isinstance(row["judge_score"], float) or row["judge_score"] is None
            assert isinstance(row["on_topic"], bool)
            serialized = json.dumps(row)
            assert "myjaem13" not in serialized and "putt" not in serialized
            assert "GusDikaOfficial" not in serialized
            assert not row["id"].startswith(("pilot-", "background-"))
        background = [json.loads(line) for line in (data_dir / "background.jsonl").read_text().splitlines()]
        assert len(background) == 30 and all(set(row) == ingest_keys for row in background)
        assert not (data_dir / "emb").exists()
        assert len(calls) == 5
        queries = collect.build_queries("sppg", collect.month_windows(collect.START, collect.END))
        assert calls[-1] == (queries, collect.MAX_PER_QUERY_DISCOVERY * len(queries))
        snapshot = file_snapshot(directory)
        output = io.StringIO()
        with redirect_stdout(output):
            collect.discover(directory, unexpected_call, unexpected_call, "test-salt", unexpected_call)
        assert state["stop_reason"] in output.getvalue()
        assert file_snapshot(directory) == snapshot


def test_auto_review_log_all_results_and_resume():
    for rejected_precision, decision in ((0.1, "reject_precision"), (0.8, "reject_cap")):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            fetch_fn, calls = discovery_fixture()

            def crash_after_log(path, fields, rows):
                assert path.name == "decisions.csv"
                assert len(collect._read_jsonl(data_dir / "review_log.jsonl")) == 2
                raise RuntimeError("interrupted after evidence was saved")

            output = io.StringIO()
            with patch.object(collect, "TOP_N_CANDIDATES", 2), \
                    patch.object(collect, "MAX_ACCEPT_PER_ROUND", 1), \
                    patch.object(collect, "pilot_stats", side_effect=[(100, 0.9), (100, rejected_precision)]), \
                    redirect_stdout(output):
                with patch.object(collect, "_append_csv", crash_after_log):
                    try:
                        collect.discover(directory, fetch_fn, fake_encode, "salt")
                    except RuntimeError as error:
                        assert str(error) == "interrupted after evidence was saved"
                    else:
                        assert False, "interrupt after logging all pilot results"
                collect.discover(directory, fetch_fn, fake_encode, "salt")
            audit = collect._read_jsonl(data_dir / "review_log.jsonl")
            assert len(audit) == 2
            assert {row["term"]: row["decision"] for row in audit} == {
                "sppg": "accept", "viral": decision,
            }
            for row in audit:
                assert set(row) == {"round", "term", "df", "lift", "novel_n", "precision_novel",
                                    "decision", "examples", "timestamp"}
                assert row["round"] == 1 and row["df"] == 40 and row["novel_n"] == 100
                assert row["lift"] >= collect.MIN_LIFT
                assert row["precision_novel"] == (0.9 if row["term"] == "sppg" else rejected_precision)
                assert len(row["examples"]) == 3 and row["timestamp"]
                assert all(row["term"] in text and len(text) <= 240 for text in row["examples"])
            assert len(calls) == 5
            assert output.getvalue().splitlines()[-1] == (
                "Discovery finished (low_gain). "
                "Next: run collect.py --pick to choose the final keywords."
            )


def test_pick_end_to_end():
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        collect._write_json(data_dir / "state.json", {
            "round": 3, "judge": {"tau": 0.42}, "keywords": collect.SEEDS + ["automatic"],
            "pending": ["automatic"], "stop_reason": "max_rounds",
        })
        rows = []
        for round_no, term, novel_n, precision, decision in [
            (3, "sppg", 100, 0.6, "reject_precision"),
            (1, "sppg", 100, 0.99, "accept"),
            (1, "bgn", 80, 0.7, "accept"),
            (2, "bgn", 90, 0.85, "reject_human"),
            (2, "langka", 19, 1.0, "reject_redundant"),
            (2, "viral", 100, 0.49, "reject_precision"),
            (2, collect.SEEDS[0], 100, 1.0, "accept"),
            (2, collect.SEEDS[1], 100, 1.0, "accept"),
        ]:
            rows.append({"round": round_no, "term": term, "df": 30, "lift": 4.5,
                         "novel_n": novel_n, "precision_novel": precision, "decision": decision,
                         "examples": [f"{term} round {round_no}"], "timestamp": "2026-10-27"})
        collect._append_jsonl(data_dir / "review_log.jsonl", rows)
        reviews = []

        def review(shortlist, title):
            assert title == "Final pick"
            assert shortlist == [rows[3], rows[0]]
            reviews.append(shortlist)
            if len(reviews) == 1:
                return ["sppg", collect.SEEDS[0]], ["dapur umum", "sppg", collect.SEEDS[1], "dapur umum"]
            return ["bgn"], []

        output = io.StringIO()
        with redirect_stdout(output):
            collect.pick(directory, review=review)
            frozen = json.loads((data_dir / "keywords_final.json").read_text())
            assert set(frozen) == {"keywords", "encoder", "tau", "params", "frozen_at", "picked_by"}
            assert frozen["keywords"] == collect.SEEDS + ["sppg", "dapur umum"]
            assert frozen["picked_by"] == "human" and frozen["tau"] == 0.42
            assert frozen["encoder"] == collect.ENCODER and frozen["params"]["SEEDS"] == collect.SEEDS
            assert frozen["frozen_at"]
            state = json.loads((data_dir / "state.json").read_text())
            assert state["stop_reason"] == "human_pick" and state["pending"] == []
            assert state["keywords"] == frozen["keywords"]
            assert state["pick"] == {"accepted": ["sppg", collect.SEEDS[0]],
                                     "added": ["dapur umum", "sppg", collect.SEEDS[1], "dapur umum"]}
            snapshot = file_snapshot(directory)
            collect.pick(directory, review=review)
            assert len(reviews) == 1 and file_snapshot(directory) == snapshot
            assert "already picked" in output.getvalue()
            collect.pick(directory, review=review, repick=True)
        assert len(reviews) == 2
        frozen = json.loads((data_dir / "keywords_final.json").read_text())
        assert frozen["keywords"] == collect.SEEDS + ["bgn"] and frozen["picked_by"] == "human"
        state = json.loads((data_dir / "state.json").read_text())
        assert state["pick"] == {"accepted": ["bgn"], "added": []}
        assert "Final keywords: " + ", ".join(collect.SEEDS + ["bgn"]) in output.getvalue()


def test_pick_resumes_saved_answer_after_crash():
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        collect._write_json(data_dir / "state.json", {"round": 1, "judge": {"tau": 0.25}})
        write_json = collect._write_json

        def crash_before_freeze(path, value):
            if path.name == "keywords_final.json":
                saved = json.loads((data_dir / "state.json").read_text())
                assert saved["pick"] == {"accepted": [], "added": ["bgn"]}
                assert saved["stop_reason"] == "human_pick" and saved["pending"] == []
                raise RuntimeError("interrupted before freezing")
            write_json(path, value)

        with redirect_stdout(io.StringIO()):
            with patch.object(collect, "_write_json", crash_before_freeze):
                try:
                    collect.pick(directory, review=lambda shortlist, title: ([], ["bgn"]))
                except RuntimeError as error:
                    assert str(error) == "interrupted before freezing"
                else:
                    assert False, "interrupt the final keyword write"
            assert not (data_dir / "keywords_final.json").exists()
            collect.pick(directory, review=unexpected_call)
        frozen = json.loads((data_dir / "keywords_final.json").read_text())
        assert frozen["keywords"] == collect.SEEDS + ["bgn"] and frozen["picked_by"] == "human"


def test_pick_requires_finished_round_one():
    with tempfile.TemporaryDirectory() as directory:
        for state in (None, {"round": 0}):
            if state is not None:
                collect._write_json(Path(directory) / "state.json", state)
            try:
                collect.pick(directory, review=unexpected_call)
            except SystemExit as error:
                assert str(error) == "run discovery first"
            else:
                assert False, "pick requires a calibrated discovery state"


def test_discover_resume_after_declining_fetch():
    with tempfile.TemporaryDirectory() as directory:
        fetch_fn, calls = discovery_fixture()
        original = collect.TOP_N_CANDIDATES
        collect.TOP_N_CANDIDATES = 2
        messages = []

        def confirm(message):
            messages.append(message)
            assert "$" in message and "cost bound" in message
            return "Background" not in message

        try:
            with redirect_stdout(io.StringIO()):
                collect.discover(directory, fetch_fn, fake_encode, "salt", confirm)
                assert len(calls) == 1
                state = json.loads((Path(directory) / "state.json").read_text())
                assert len(state["collecting"]["seed_rows"]) == 40
                assert not (Path(directory) / "posts.jsonl").exists()
                collect.discover(directory, fetch_fn, fake_encode, "salt",
                                 lambda message: not message.startswith("Pilots:"))
                assert len(calls) == 2
                state = json.loads((Path(directory) / "state.json").read_text())
                assert state["round"] == 1
                assert state["pilots"]["results"] == []
                assert all(len(candidate) == 3 for candidate in state["pilots"]["candidates"])
                posts_before = (Path(directory) / "posts.jsonl").read_bytes()
                collect.discover(directory, fetch_fn, fake_encode, "salt")
            assert len(messages) == 2
            assert len(calls) == 5
            assert (Path(directory) / "posts.jsonl").read_bytes().startswith(posts_before)
            assert (Path(directory) / "keywords_final.json").exists()
        finally:
            collect.TOP_N_CANDIDATES = original


def test_discover_pilot_batch_resume():
    with tempfile.TemporaryDirectory() as directory:
        fetch_fn, calls = discovery_fixture()
        messages = []
        failed = False

        def fetch_with_crash(queries, max_items):
            nonlocal failed
            if max_items == collect.PILOT_N and queries[0].startswith("viral ") and not failed:
                failed = True
                raise RuntimeError("pilot interrupted")
            return fetch_fn(queries, max_items)

        def confirm(message):
            messages.append(message)
            return True

        with patch.object(collect, "TOP_N_CANDIDATES", 2), redirect_stdout(io.StringIO()):
            try:
                collect.discover(directory, fetch_with_crash, fake_encode, "salt", confirm)
            except RuntimeError as error:
                assert str(error) == "pilot interrupted"
            else:
                assert False, "interrupt the second pilot after confirming the batch"
            state = json.loads((Path(directory) / "state.json").read_text())
            assert state["pilots"]["confirmed"] is True
            assert [row["term"] for row in state["pilots"]["results"]] == ["sppg"]
            collect.discover(directory, fetch_with_crash, fake_encode, "salt", confirm)
        pilot_messages = [message for message in messages if message.startswith("Pilots:")]
        assert len(pilot_messages) == 1
        assert "2 candidates" in pilot_messages[0]
        assert f"${collect.cost(2 * collect.PILOT_N):.6f}" in pilot_messages[0]
        assert len(calls) == 5


def test_discover_with_review():
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        R, B = calibration_texts()
        windows = collect.month_windows(collect.START, collect.END)
        seed_queries = [q for term in collect.SEEDS for q in collect.build_queries(term, windows)]
        food = mask_terms(" ".join(R), collect.SEEDS)
        calls, messages, reviews = [], [], []

        def fetch_fn(queries, max_items):
            calls.append((queries, max_items))
            if queries == seed_queries:
                assert max_items == collect.MAX_PER_QUERY_DISCOVERY * len(queries)
                return [raw_tweet(f"seed-{i}", (R[i % len(R)] + " ") * 10 +
                                  " dan sppg dan bgn dan viral dan langka @nama", i % len(queries))
                        for i in range(40)]
            if queries[0].startswith(collect.BACKGROUND_QUERY + " "):
                assert len(queries) == 1 and max_items == collect.BACKGROUND_N
                return [raw_tweet(f"background-{i}", B[i % len(B)]) for i in range(30)]
            if max_items == collect.PILOT_N:
                assert len(queries) == 1
                term = queries[0].split(" since:")[0]
                text = (R[0] if term == "langka" else B[0] if term == "viral" else food)
                return [raw_tweet(f"pilot-{term}-{i}", term + " " + text) for i in range(100)]
            assert len(queries) == 2 * len(windows)
            assert max_items == collect.MAX_PER_QUERY_DISCOVERY * len(queries)
            return [raw_tweet("round2-new", food)]

        def confirm(message):
            messages.append(message)
            return True

        def review(shortlist):
            reviews.append(copy.deepcopy(shortlist))
            assert {row["term"] for row in shortlist} == {"bgn", "sppg"}
            assert [row["precision_novel"] for row in shortlist] == sorted(
                [row["precision_novel"] for row in shortlist], reverse=True
            )
            assert all(row["decision"] == "review" and len(row["examples"]) == 3 for row in shortlist)
            return [shortlist[0]["term"]], ["dapur umum"]

        append_csv = collect._append_csv

        def crash_after_review(path, fields, rows):
            if path.name == "decisions.csv":
                saved = json.loads((data_dir / "state.json").read_text())
                assert saved["pilots"]["review"] == [[reviews[0][0]["term"]], ["dapur umum"]]
                raise RuntimeError("review saved; interrupted before decisions")
            append_csv(path, fields, rows)

        with patch.object(collect, "TOP_N_CANDIDATES", 4), \
                patch.object(collect, "MAX_ACCEPT_PER_ROUND", 0), redirect_stdout(io.StringIO()):
            with patch.object(collect, "_append_csv", crash_after_review):
                try:
                    collect.discover(directory, fetch_fn, fake_encode, "salt", confirm, review)
                except RuntimeError as error:
                    assert str(error) == "review saved; interrupted before decisions"
                else:
                    assert False, "interrupt after persisting the review answer"
            collect.discover(directory, fetch_fn, fake_encode, "salt", confirm, review)
            snapshot = file_snapshot(directory)
            collect.discover(directory, unexpected_call, unexpected_call, "salt", unexpected_call, review)
            assert file_snapshot(directory) == snapshot
        assert len(reviews) == 1
        pilot_messages = [message for message in messages if message.startswith("Pilots:")]
        assert len(pilot_messages) == 1 and "4 candidates" in pilot_messages[0]
        assert f"${collect.cost(4 * collect.PILOT_N):.6f}" in pilot_messages[0]
        with (data_dir / "decisions.csv").open() as stream:
            reader = csv.DictReader(stream)
            decisions = list(reader)
            assert reader.fieldnames == ["round", "term", "df", "lift", "novel_n", "precision_novel",
                                         "tau", "b0", "decision", "pilot_cost_usd", "timestamp"]
        accepted = reviews[0][0]["term"]
        rejected = reviews[0][1]["term"]
        by_term = {row["term"]: row for row in decisions}
        assert len(decisions) == 5
        assert {term: row["decision"] for term, row in by_term.items()} == {
            accepted: "accept_human", rejected: "reject_human", "dapur umum": "human_added",
            "langka": "reject_redundant", "viral": "reject_precision",
        }
        assert all(float(by_term[term]["lift"]) >= collect.MIN_LIFT for term in [accepted, rejected])
        assert by_term["dapur umum"]["lift"] == "" and by_term["dapur umum"]["df"] == "0"
        audit = collect._read_jsonl(data_dir / "review_log.jsonl")
        assert len(audit) == 2 and {row["term"] for row in audit} == {accepted, rejected}
        for row in audit:
            assert set(row) == {"round", "term", "df", "lift", "novel_n", "precision_novel",
                                "decision", "examples", "timestamp"}
            assert row["df"] == int(by_term[row["term"]]["df"])
            assert row["lift"] == float(by_term[row["term"]]["lift"])
            assert row["novel_n"] == int(by_term[row["term"]]["novel_n"])
            assert row["precision_novel"] == float(by_term[row["term"]]["precision_novel"])
            assert row["decision"] == by_term[row["term"]]["decision"]
            assert len(row["examples"]) == 3
            assert all(len(text) <= 240 and mask_terms(text, collect.SEEDS) == text for text in row["examples"])
        frozen = json.loads((data_dir / "keywords_final.json").read_text())
        assert frozen["keywords"] == collect.SEEDS + [accepted, "dapur umum"]
        state = json.loads((data_dir / "state.json").read_text())
        assert state["history"][1]["keywords"] == [accepted, "dapur umum"]
        assert state["stop_reason"] == "low_gain"
        assert len(calls) == 7


def test_discover_review_accepts_none():
    with tempfile.TemporaryDirectory() as directory:
        fetch_fn, calls = discovery_fixture()
        reviews = []

        def review(shortlist):
            reviews.append(shortlist)
            return [], []

        with patch.object(collect, "TOP_N_CANDIDATES", 2), redirect_stdout(io.StringIO()):
            collect.discover(directory, fetch_fn, fake_encode, "salt", review=review)
        assert len(reviews) == 1 and len(calls) == 4
        state = json.loads((Path(directory) / "state.json").read_text())
        assert state["stop_reason"] == "no_accepted" and state["pending"] == []
        assert state["keywords"] == collect.SEEDS


def test_discover_max_round_skips_pilots():
    with tempfile.TemporaryDirectory() as directory:
        original = collect.MAX_ROUNDS
        collect.MAX_ROUNDS = 1
        try:
            calls = run_discovery(directory)
            assert len(calls) == 2
            state = json.loads((Path(directory) / "state.json").read_text())
            assert state["stop_reason"] == "max_rounds"
            assert json.loads((Path(directory) / "keywords_final.json").read_text())["keywords"] == collect.SEEDS
        finally:
            collect.MAX_ROUNDS = original


def test_discover_no_candidates_and_acceptance_cap():
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(collect, "TOP_N_CANDIDATES", 0):
            fetch_fn, calls = discovery_fixture()
            with redirect_stdout(io.StringIO()):
                collect.discover(directory, fetch_fn, fake_encode, "salt")
        assert len(calls) == 2
        state = json.loads((Path(directory) / "state.json").read_text())
        assert state["stop_reason"] == "no_accepted"
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(collect, "MAX_ACCEPT_PER_ROUND", 1), \
                patch.object(collect, "pilot_stats", return_value=(100, 0.9)):
            run_discovery(directory)
        with (Path(directory) / "decisions.csv").open() as stream:
            decisions = {row["term"]: row["decision"] for row in csv.DictReader(stream)}
        assert decisions == {"sppg": "accept", "viral": "reject_cap"}
        frozen = json.loads((Path(directory) / "keywords_final.json").read_text())
        assert frozen["keywords"] == collect.SEEDS + ["sppg"]


def test_discover_crash_after_append_keeps_gain_and_deduplication():
    with tempfile.TemporaryDirectory() as directory:
        fetch_fn, calls = discovery_fixture()
        append = collect._append_jsonl

        def crash_after_append(path, rows):
            append(path, rows)
            if path.name == "posts.jsonl" and any(row["id"] == "round2-new" for row in rows):
                raise RuntimeError("simulated crash")

        with patch.object(collect, "TOP_N_CANDIDATES", 2), redirect_stdout(io.StringIO()):
            with patch.object(collect, "_append_jsonl", crash_after_append):
                try:
                    collect.discover(directory, fetch_fn, fake_encode, "salt")
                except RuntimeError as error:
                    assert str(error) == "simulated crash"
                else:
                    assert False, "the crash must occur after round 2 was appended"
            calibration_before = (Path(directory) / "calibration.npz").read_bytes()
            collect.discover(directory, fetch_fn, fake_encode, "salt")
        assert len(calls) == 6
        rows = [json.loads(line) for line in (Path(directory) / "posts.jsonl").read_text().splitlines()]
        assert len(rows) == len({row["id"] for row in rows}) == 41
        state = json.loads((Path(directory) / "state.json").read_text())
        assert state["history"][1]["new_unique"] == 1
        assert state["history"][1]["new_on_topic"] == 1
        assert state["history"][1]["gain"] == 1 / 40
        assert (Path(directory) / "calibration.npz").read_bytes() == calibration_before


def test_smoke_cost_first_without_encoder():
    output = io.StringIO()
    item = sample_tweet()
    with patch.dict(os.environ, APIFY_TOKEN="secret-token", USER_HASH_SALT="secret-salt"), \
            patch.object(collect, "fetch", return_value=[item]) as mocked_fetch, \
            patch.object(collect, "rotating_fetch", return_value=collect.rotating_fetch(
                ["secret-token"], fetch_one=mocked_fetch, rng=random.Random(0)
            )) as mocked_rotation, \
            patch.object(collect.judge, "load_encoder", unexpected_call), \
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


def test_final_collection():
    with tempfile.TemporaryDirectory() as directory:
        run_discovery(directory)
        data_dir = Path(directory)
        original = collect.START, collect.END, collect.MAX_PER_QUERY_FINAL
        collect.START, collect.END, collect.MAX_PER_QUERY_FINAL = "2025-01-01", "2025-03-17", 2
        calls = []
        R, B = calibration_texts()

        def fetch_fn(queries, max_items):
            calls.append((queries, max_items))
            assert len(queries) == 3 and max_items == 6
            term = queries[0].split(" since:")[0]
            return [raw_tweet("shared", R[0], 0), raw_tweet(term + "-0", R[1], 0),
                    raw_tweet(term + "-1", B[0], 1), raw_tweet(term + "-2", "MBG", 2)]

        try:
            with redirect_stdout(io.StringIO()):
                collect.final(directory, fetch_fn, fake_encode, "salt", lambda message: True,
                              allow_auto=True)
            assert len(calls) == 3
            rows = [json.loads(line) for line in (data_dir / "final_posts.jsonl").read_text().splitlines()]
            assert len(rows) == len({row["id"] for row in rows}) == 10
            assert any(row["judge_score"] is None and row["on_topic"] is False for row in rows)
            assert any(row["text"] == B[0] and row["on_topic"] is False for row in rows)
            assert all("since:" in row["query"] for row in rows)
            with (data_dir / "collection_log.csv").open() as stream:
                reader = csv.DictReader(stream)
                logs = list(reader)
                assert reader.fieldnames == ["keyword", "month", "fetched", "capped"]
            assert len(logs) == 9
            for log in logs:
                assert int(log["fetched"]) == (2 if log["month"] == "2025-01" else 1)
                assert log["capped"] == ("True" if log["month"] == "2025-01" else "False")
            snapshot = file_snapshot(directory)
            with redirect_stdout(io.StringIO()):
                collect.final(directory, unexpected_call, unexpected_call, "salt", unexpected_call,
                              allow_auto=True)
            assert file_snapshot(directory) == snapshot
        finally:
            collect.START, collect.END, collect.MAX_PER_QUERY_FINAL = original


def test_final_requires_human_pick():
    with tempfile.TemporaryDirectory() as directory:
        run_discovery(directory)
        with patch.object(collect.judge, "load_calibration", unexpected_call):
            try:
                collect.final(directory, unexpected_call, unexpected_call, "salt", unexpected_call)
            except SystemExit as error:
                assert str(error) == (
                    "keywords were chosen automatically; run --pick first (or pass --allow-auto)"
                )
            else:
                assert False, "automatic keywords require an explicit opt-in"
        with redirect_stdout(io.StringIO()):
            collect.pick(directory, review=lambda shortlist, title: ([], []))
            collect.final(directory, unexpected_call, unexpected_call, "salt", lambda message: False)


def test_wilson():
    low, high = validate.wilson(80, 100)
    assert abs(low - 0.711) < 0.005 and abs(high - 0.867) < 0.005
    assert validate.wilson(0, 10)[0] == 0
    assert validate.wilson(10, 10)[1] == 1
    try:
        validate.wilson(0, 0)
    except ValueError:
        pass
    else:
        assert False, "Wilson interval requires at least one label"


def test_sample_stratum():
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        rows = [{"id": str(i), "text": ("MBG " if i % 2 == 0 else "")
                 + f"menu sehat, siswa\nsekolah {i}", "user_hash": "private",
                 "bio": "private", "followers": 42, "on_topic": False} for i in range(150)]
        rows += [{"id": "phrase", "text": "Makan Bergizi Gratis untuk siswa"},
                 {"id": "hashtag", "text": "#MakanBergiziGratis untuk siswa"}]
        collect._append_jsonl(data_dir / "final_posts.jsonl", rows)
        output = io.StringIO()
        with redirect_stdout(output):
            validate.sample(directory, n=20, seed=7)
        assert "75 rows" in output.getvalue() and "sampled 20" in output.getvalue()
        path = data_dir / "validation_sample.csv"
        before = path.read_bytes()
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == ["id", "text", "label"]
            sampled = list(reader)
        assert len(sampled) == len({row["id"] for row in sampled}) == 20
        assert all("mbg" not in row["text"].lower() and row["label"] == "" for row in sampled)
        assert all(row["text"] == rows[int(row["id"])]["text"] for row in sampled)
        assert b"private" not in before and b"followers" not in before
        with redirect_stdout(io.StringIO()):
            validate.sample(directory, n=20, seed=7)
            assert path.read_bytes() == before
            validate.sample(directory, n=100, seed=7)
        with path.open(newline="", encoding="utf-8") as stream:
            all_novel = list(csv.DictReader(stream))
        assert len(all_novel) == 75
        assert {row["id"] for row in all_novel} == {str(i) for i in range(1, 150, 2)}
        sampled[0]["label"] = "1"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "text", "label"])
            writer.writeheader()
            writer.writerows(sampled)
        before = path.read_bytes()
        try:
            validate.sample(directory)
        except SystemExit as error:
            assert "overwrite" in str(error) and "labeled" in str(error)
        else:
            assert False, "a labeled sample must not be overwritten"
        assert path.read_bytes() == before


def validation_fixture(directory):
    data_dir = Path(directory)
    R, B = calibration_texts()
    collect._append_jsonl(data_dir / "posts.jsonl", [{"text": text} for text in R + B[:2]])
    collect._append_jsonl(data_dir / "background.jsonl", [{"text": text} for text in B])
    cal = calibrate(R, B, fake_encode, collect.SEEDS)
    save_calibration(cal, data_dir / "calibration.npz")
    rows = [{"id": f"positive-{i}", "text": mask_terms(text, collect.SEEDS), "label": "1"}
            for i, text in enumerate(R)]
    rows += [{"id": f"negative-{i}", "text": text, "label": "0"}
             for i, text in enumerate(B[:8])]
    rows += [{"id": f"unsure-{i}", "text": text, "label": ""} for i, text in enumerate(B[:2])]
    with (data_dir / "validation_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return R, B, cal


def test_score_labels_end_to_end():
    with tempfile.TemporaryDirectory() as directory:
        _, _, cal = validation_fixture(directory)
        before = file_snapshot(directory)
        output = io.StringIO()
        with patch.object(validate.judge, "load_encoder", unexpected_call), \
                patch.object(validate.judge, "calibrate", unexpected_call), redirect_stdout(output):
            report = validate.score_labels(directory, encode=fake_encode)
            explicit = validate.score_labels(directory, encoder_name=collect.ENCODER, encode=fake_encode)
        assert report == explicit
        assert report["n_labeled"] == 20 and report["n_skipped"] == 2
        assert report["precision"] == 12 / 20
        assert report["ci"][0] <= report["precision"] <= report["ci"][1]
        assert report["judge_auc"] >= 0.9
        assert report["judge_precision"] == report["judge_recall"] == 1.0
        assert report["n_nan"] == 0
        assert report["encoder"] == collect.ENCODER and report["tau"] == cal["tau"]
        assert "Labeled: 20" in output.getvalue() and "skipped (unsure): 2" in output.getvalue()
        assert "Wilson 95% CI" in output.getvalue() and "Judge AUC" in output.getvalue()
        assert "precision/recall at tau=" in output.getvalue() and collect.ENCODER in output.getvalue()
        assert "AUC differences under ~0.05 are noise at n=100" in output.getvalue()
        assert file_snapshot(directory) == before


def test_score_labels_other_encoder():
    with tempfile.TemporaryDirectory() as directory:
        R, B, _ = validation_fixture(directory)
        before = file_snapshot(directory)
        output = io.StringIO()
        with patch.object(validate.judge, "load_encoder", unexpected_call), \
                patch.object(validate.judge, "load_calibration", unexpected_call), \
                patch.object(validate.judge, "calibrate", wraps=calibrate) as recalibrate, \
                redirect_stdout(output):
            report = validate.score_labels(directory, encoder_name="other-encoder", encode=fake_encode)
        recalibrate.assert_called_once_with(R, B, fake_encode, collect.SEEDS)
        assert report["encoder"] == "other-encoder" and report["judge_auc"] >= 0.9
        assert "approximate round-1 posts" in output.getvalue()
        assert "seed-matching" in output.getvalue() and "other-encoder" in output.getvalue()
        assert file_snapshot(directory) == before


def test_score_labels_nan_scores():
    with tempfile.TemporaryDirectory() as directory:
        R, B, _ = validation_fixture(directory)
        path = Path(directory) / "validation_sample.csv"
        path.write_text(f"id,text,label\np,{R[0]},1\nn,{B[0]},0\n"
                        "short-positive,MBG,1\nshort-negative,dua kata,0\nunsure,skip,\n",
                        encoding="utf-8")
        output = io.StringIO()
        with patch.object(validate.judge, "load_encoder", unexpected_call), redirect_stdout(output):
            report = validate.score_labels(directory, encode=fake_encode)
        assert report["n_labeled"] == 4 and report["n_skipped"] == 1
        assert report["n_nan"] == 2 and report["precision"] == 0.5
        assert report["judge_auc"] == 1.0 and report["judge_precision"] == 1.0
        assert report["judge_recall"] == 0.5
        assert "NaN scores: 2" in output.getvalue()


def test_score_labels_empty_groups_without_model():
    with tempfile.TemporaryDirectory() as directory:
        _, B, _ = validation_fixture(directory)
        path = Path(directory) / "validation_sample.csv"
        with patch.object(validate.judge, "load_encoder", unexpected_call):
            path.write_text(f"id,text,label\nnegative,{B[0]},0\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                report = validate.score_labels(directory, encode=fake_encode)
            assert np.isnan(report["judge_auc"]) and report["precision"] == 0
            assert report["judge_precision"] == report["judge_recall"] == 0
            path.write_text("id,text,label\nunsure,skip,\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                report = validate.score_labels(directory)
            assert report["n_labeled"] == 0 and report["n_skipped"] == 1
            assert np.isnan(report["precision"]) and all(np.isnan(value) for value in report["ci"])
            assert np.isnan(report["judge_auc"]) and report["n_nan"] == 0
            assert report["judge_precision"] == report["judge_recall"] == 0
            path.write_text("id,text,label\n" + "short,dua kata,0\n" * 200, encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                report = validate.score_labels(directory)
            assert report["n_labeled"] == report["n_nan"] == 200
            assert np.isnan(report["judge_auc"])
            assert "AUC differences" not in output.getvalue()


def test_bad_label():
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "validation_sample.csv").write_text(
            "id,text,label\nbad-row,menu makanan sehat,yes\n", encoding="utf-8")
        with patch.object(validate.judge, "load_encoder", unexpected_call):
            try:
                validate.score_labels(directory, encode=fake_encode)
            except SystemExit as error:
                assert "bad-row" in str(error) and "yes" in str(error)
            else:
                assert False, "invalid labels must name the row id"


def test_validate_cli():
    assert validate.DATA_DIR == "data"
    with patch.object(validate, "sample") as sample_fn, \
            patch.object(validate, "score_labels") as score_fn:
        validate.main(["sample"])
        sample_fn.assert_called_once_with("data")
        validate.main(["score"])
        score_fn.assert_called_once_with("data", None)
        score_fn.reset_mock()
        validate.main(["score", "--encoder", "other-encoder"])
        score_fn.assert_called_once_with("data", "other-encoder")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
    print("ok")
