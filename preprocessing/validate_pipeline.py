#!/usr/bin/env python3
"""Validasi deterministik output preprocessing sebelum BERTopic."""

from __future__ import annotations

import csv
import hashlib
import json
import py_compile
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

import preprocess_twitter as pipeline
import merge_raw


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    results: dict[str, dict] = {}

    for filename in ["preprocess_twitter.py", "validate_pipeline.py", "merge_raw.py"]:
        py_compile.compile(str(ROOT / filename), doraise=True)
    results["python_syntax"] = {"status": "passed"}

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        old, new, out = [folder / name for name in ["old.jsonl", "new.jsonl", "raw_data_MBG.jsonl"]]
        old.write_text(json.dumps({"id": "1", "text": "baris\u2028tetap satu 👍"}, ensure_ascii=False) + "\n", encoding="utf-8")
        new.write_text(json.dumps({"id": "2", "pull_batch": "slot_sample"}) + "\n", encoding="utf-8")
        out.write_bytes(old.read_bytes())
        backup = folder / "raw_data_MBG_batch1_month_wide.jsonl"
        merge_raw.merge_raw(old, new, out)
        merged = out.read_bytes()
        check(backup.read_bytes() == old.read_bytes(), "Arsip batch 1 berubah")
        check([row["pull_batch"] for row in merge_raw.read_jsonl(out)] == ["month_wide", "slot_sample"], "Urutan/metadata merge salah atau U+2028 memecah record")
        merge_raw.merge_raw(old, new, out)
        check(out.read_bytes() == merged and backup.read_bytes() == old.read_bytes(), "Merge ulang tidak identik atau menimpa arsip")
        for duplicate_rows in [[{"id": "1"}], [{"id": "2"}, {"id": "2"}]]:
            new.write_text("".join(json.dumps(row) + "\n" for row in duplicate_rows), encoding="utf-8")
            try:
                merge_raw.merge_raw(old, new, out)
            except ValueError:
                pass
            else:
                raise AssertionError("ID ganda harus menghentikan merge")
            check(out.read_bytes() == merged and backup.read_bytes() == old.read_bytes(), "Merge ID ganda mengubah file")
        untouched = folder / "untouched"
        untouched.mkdir()
        try:
            merge_raw.merge_raw(old, new, untouched / "raw_data_MBG.jsonl")
        except ValueError:
            pass
        check(not list(untouched.iterdir()), "Merge ID ganda menulis file baru")
    results["raw_merge_cases"] = {"status": "passed"}

    filenames = ["posts_rq2_rq3.csv", "topic_documents_rq1.csv", "manual_quality_audit.csv", "preprocessing_report.json", "topic_stopwords.txt", "posts_preprocessed.jsonl"]
    with tempfile.TemporaryDirectory() as directory:
        fresh = Path(directory)
        subprocess.run([
            sys.executable, str(ROOT / "preprocess_twitter.py"),
            "--posts-output", str(fresh / filenames[0]),
            "--topic-output", str(fresh / filenames[1]),
            "--audit-output", str(fresh / filenames[2]),
            "--report", str(fresh / filenames[3]),
            "--stopwords-output", str(fresh / filenames[4]),
            "--json-output", str(fresh / filenames[5]),
        ], check=True, capture_output=True, text=True)
        fresh_report = json.loads((fresh / filenames[3]).read_text(encoding="utf-8"))
        # Samakan hanya path output sementara agar hash report dapat dibandingkan.
        fresh_report["configuration"]["stopwords_output"] = str(OUTPUT / filenames[4])
        (fresh / filenames[3]).write_text(json.dumps(fresh_report, ensure_ascii=False, indent=2), encoding="utf-8")
        for filename in filenames:
            check(
                (OUTPUT / filename).is_file()
                and hashlib.sha256((fresh / filename).read_bytes()).digest() == hashlib.sha256((OUTPUT / filename).read_bytes()).digest(),
                "Output usang; jalankan preprocess_twitter.py ulang",
            )
    results["output_freshness"] = {"status": "passed", "files": filenames}

    raw = pipeline.load_data(ROOT / "raw_data_MBG.jsonl")
    posts = pd.read_csv(OUTPUT / "posts_rq2_rq3.csv", dtype={"id": "string", "source_month": "string", "window_capped": "boolean"})
    topic = pd.read_csv(OUTPUT / "topic_documents_rq1.csv", dtype={"id": "string", "source_month": "string", "window_capped": "boolean"})
    audit = pd.read_csv(OUTPUT / "manual_quality_audit.csv", dtype={"id": "string"})
    retention = json.loads((OUTPUT / "preprocessing_report.json").read_text())["retention"]
    check(retention["input_rows"] == len(raw), "Retention input tidak cocok")
    results["raw_records"] = {"status": "passed", "rows": len(raw)}
    check(retention["all_valid_posts_rows"] == len(posts), "Retention all-valid tidak cocok")
    check(retention["posts_rq2_rq3_rows"] == len(posts), "Retention RQ2/RQ3 tidak cocok")
    check(retention["topic_rows_after_exact_text_deduplication"] == len(topic), "Retention RQ1 tidak cocok")
    check(retention["window_capped_true_rows"] == int(posts["window_capped"].eq(True).sum()), "Retention capped True tidak cocok")
    check(retention["window_capped_false_rows"] == int(posts["window_capped"].eq(False).sum()), "Retention capped False tidak cocok")
    check(retention["window_capped_unknown_rows"] == int(posts["window_capped"].isna().sum()), "Retention capped unknown tidak cocok")
    results["retention_stages"] = {"status": "passed", **retention}

    check(len(audit) == 150, "Audit tidak berisi 150 sampel")
    check(audit["text"].notna().all() and audit["text_normalized"].notna().all(), "Audit memiliki teks kosong")
    check(audit["id"].tolist() == posts.sample(n=150, random_state=42)["id"].tolist(), "Sampel audit tidak cocok dengan seed 42")
    results["raw_processed_sample"] = {
        "status": "technical_passed_human_review_pending",
        "rows": len(audit),
        "human_review_completed": int(audit["review_status"].eq("selesai").sum()),
    }

    examples = {
        "slang": pipeline.clean_normalized("gmn knp emang udh gakkk"),
        "emoji_complex": pipeline.clean_normalized("Bagus 👍🏽 keluarga 👨‍👩‍👧‍👦"),
        "hashtag": pipeline.clean_normalized("#MakanBergiziGratis"),
        "code_mixing": pipeline.clean_normalized("Aku prepare presentation besok"),
    }
    check(examples["slang"] == "bagaimana kenapa memang sudah tidak", "Normalisasi slang gagal")
    check("emoji_" in examples["emoji_complex"] and "👍" not in examples["emoji_complex"], "Emoji kompleks gagal")
    check(examples["hashtag"] == "makan bergizi gratis", "Segmentasi hashtag gagal")
    check("prepare presentation" in examples["code_mixing"], "Code-mixing tidak dipertahankan")
    for text, expected in {"massa": "massa", "Allah": "allah", "telattt": "telat", "baguuuus": "bagus"}.items():
        check(pipeline.clean_normalized(text) == expected, f"Normalisasi repetisi {text} gagal")
    check(pipeline.clean_normalized("Bagus 👍🏽") == pipeline.clean_normalized("Bagus 👍"), "Skin tone emoji tidak disatukan")
    check("emoji_family_man_woman_girl_boy" in examples["emoji_complex"], "Urutan ZWJ emoji berubah")
    check("dr" in pipeline.clean_normalized("dr Tan bilang ok").split(), "Singkatan dr berubah")
    check(pipeline.clean_normalized("#KendalaDanaSPPGSolo") == "kendala dana sppg solo", "Segmentasi akronim hashtag gagal")
    check(pipeline.segment_hashtag("awasimbg") == "awasi mbg", "Koreksi hashtag gagal")
    topic_example = pipeline.clean_topic("Cek https://t.co/x @pengguna MBG bagus")
    check("[url]" not in topic_example and "[user]" not in topic_example and "mbg" in topic_example.split(), "Placeholder/query pada text_topic salah")
    check(pipeline.clean_embed("@a @b Bagus 👍 https://t.co/x kata @c") == "Bagus 👍 kata @c", "Pembersihan embedding mengubah isi kalimat")
    results["transformation_cases"] = {"status": "passed", "examples": examples}

    fixture = pd.DataFrame({
        "id": ["late", "early", "link", "mention", "invalid", "early", "tie"],
        "text": ["@a MBG bagus https://t.co/late", "@a mbg BAGUS https://t.co/early", "https://t.co/only", "@a @b https://t.co/x", "@a MBG bagus", "teks lain", "@a mbg bagus"],
        "created_at": ["2025-01-02T00:00:00Z", "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "invalid", "2025-01-03T00:00:00Z", "2025-01-01T00:00:00Z"],
        "bio": ["profil"] * 7,
    })
    prepared, _ = pipeline.prepare_data(fixture)
    fixture_posts, fixture_topic, _ = pipeline.build_outputs(prepared)
    check(len(fixture_posts) == 5 and fixture_topic["id"].tolist() == ["early"], "Deduplikasi tidak memilih posting layak paling awal secara stabil")
    check(fixture_topic["dup_group_size"].tolist() == [3], "Ukuran kelompok memasukkan posting tidak valid")
    check(fixture_posts.loc[fixture_posts["id"].isin(["link", "mention"]), "topic_document_id"].isna().all(), "Embedding kosong harus boleh tanpa dokumen topik")
    check(prepared.loc[prepared["id"].eq("link"), "duplicate_text_group"].iloc[0] == pipeline.text_group_id("https://t.co/only"), "Fallback kunci URL kosong gagal")
    check(prepared["source_keyword"].isna().all() and "bio" not in prepared, "Query opsional/privasi gagal")
    check(prepared["source_month"].isna().all() and str(prepared["source_month"].dtype) == "string", "source_month tanpa query harus string missing")
    check(prepared["window_capped"].isna().all(), "Window tanpa query harus unknown")
    check(prepared["pull_batch"].eq("month_wide").all(), "Input tanpa pull_batch harus month_wide")
    check(prepared["is_invalid_date"].iloc[-1], "Tanggal NaT tidak berada paling akhir")
    results["deduplication_cases"] = {"status": "passed"}

    window_fixture = pd.DataFrame({
        "id": ["late", "early", "uncapped", "case", "month", "no_since", "missing_query"],
        "text": ["teks sama", "teks sama", "teks lain", "huruf besar", "bulan lain", "tanpa since", ""],
        "created_at": ["2025-02-02T00:00:00Z", "2025-01-01T00:00:00Z"] + ["2025-02-01T00:00:00Z"] * 5,
        "query": [
            "mbg since:2025-02-01_00:00:00_UTC until:2025-03-01_00:00:00_UTC",
            "mbg since:2025-01-01_00:00:00_UTC until:2025-02-01_00:00:00_UTC",
            '"makan bergizi gratis" since:2025-02-01_00:00:00_UTC',
            "MBG since:2025-01-01_00:00:00_UTC",
            "mbg since:2025-03-01_00:00:00_UTC",
            "mbg", pd.NA,
        ],
    })
    with tempfile.TemporaryDirectory() as directory:
        collection_log = Path(directory) / "collection_log.csv"
        with collection_log.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["keyword", "month", "fetched", "capped"])
            writer.writerows([
                ["mbg", "2025-01", 300, "True"],
                ["mbg", "2025-02", 25, "False"],
                ["makan bergizi gratis", "2025-02", 10, "False"],
            ])
        prepared, _ = pipeline.prepare_data(window_fixture, collection_log)
        flags = prepared.set_index("id")["window_capped"]
        check(str(flags.dtype) == "boolean", "window_capped bukan nullable boolean")
        check(flags.loc["early"] == True and flags.loc[["late", "uncapped"]].eq(False).all(), "Pemetaan capped True/False gagal")
        check(flags.loc[["case", "month", "no_since", "missing_query"]].isna().all(), "Window tidak dikenal harus missing; keyword harus cocok persis")
        months = prepared.set_index("id")["source_month"]
        check(str(months.dtype) == "string" and months.loc["early"] == "2025-01" and months.loc["late"] == "2025-02", "Ekstraksi source_month gagal")
        check(months.loc[["no_since", "missing_query"]].isna().all(), "source_month tanpa since/query harus missing")
        fixture_posts, fixture_topic, stages = pipeline.build_outputs(prepared)
        representative = fixture_topic.set_index("id").loc["early"]
        check("late" not in set(fixture_topic["id"]) and representative["source_month"] == "2025-01" and representative["window_capped"] == True, "Metadata dokumen topik bukan dari posting paling awal")
        check([stages[f"window_capped_{label}_rows"] for label in ["true", "false", "unknown"]] == [1, 2, 3], "Hitungan capped harus memakai posting valid")
        check(not {"source_month", "window_capped"}.intersection(pipeline.make_audit_sample(fixture_posts, 2, 42).columns), "Metadata window tidak boleh masuk audit")
        missing, _ = pipeline.prepare_data(window_fixture, Path(directory) / "missing.csv")
        check(str(missing["window_capped"].dtype) == "boolean" and missing["window_capped"].isna().all(), "Log hilang harus menghasilkan boolean all-NA")
    results["collection_window_cases"] = {"status": "passed"}

    batch_fixture = pd.DataFrame([
        {
            "id": "slot", "text": "teks slot", "created_at": "2025-01-02T00:00:00Z",
            "query": 'bgn -mbg -"makan bergizi gratis" -sppg since_time:1735776000 until_time:1735797600',
            "source_keyword": "bgn", "source_month": "2025-01", "pull_batch": "slot_sample",
            "slot_since": "2025-01-02T00:00:00+00:00", "slot_until": "2025-01-02T06:00:00+00:00",
        },
        {
            "id": "month", "text": "teks bulanan", "created_at": "2025-01-01T00:00:00Z",
            "query": '"makan bergizi gratis" since:2025-01-01_00:00:00_UTC until:2025-02-01_00:00:00_UTC',
        },
        {
            "id": "invalid", "text": "", "created_at": "2025-01-01T00:00:00Z",
            "pull_batch": "slot_sample",
        },
    ])
    prepared, _ = pipeline.prepare_data(batch_fixture)
    by_id = prepared.set_index("id")
    check(by_id.loc["slot", "source_keyword"] == "bgn" and by_id.loc["slot", "source_month"] == "2025-01", "Metadata slot_sample tertimpa parsing query")
    check(by_id.loc["month", "source_keyword"] == "makan bergizi gratis" and by_id.loc["month", "source_month"] == "2025-01", "Metadata month_wide tidak diparse saat missing")
    check(by_id.loc["month", "pull_batch"] == "month_wide" and by_id.loc["slot", "pull_batch"] == "slot_sample", "Fallback pull_batch salah")
    check(by_id.loc["slot", ["slot_since", "slot_until"]].tolist() == [batch_fixture.iloc[0][column] for column in ["slot_since", "slot_until"]], "Batas slot berubah")
    check(by_id.loc["month", ["slot_since", "slot_until"]].isna().all(), "month_wide memiliki batas slot")
    _, _, stages = pipeline.build_outputs(prepared)
    check([stages[f"pull_batch_{batch}_rows"] for batch in ["month_wide", "slot_sample"]] == [1, 1], "Hitungan batch memasukkan posting tidak valid")
    results["batch_metadata_cases"] = {"status": "passed"}

    for name, frame, columns in [
        ("posts", posts, ["text_minimal"]),
        ("topic", topic, ["text_minimal", "text_normalized", "text_embed"]),
    ]:
        for column in columns:
            empty = frame[column].fillna("").str.strip().eq("").sum()
            check(empty == 0, f"{name}.{column} memiliki {empty} teks kosong")
    results["non_empty_cleaned_text"] = {"status": "passed"}

    ordered_raw = raw.assign(created_at=pd.to_datetime(raw["created_at"], utc=True, errors="coerce")).sort_values("created_at", kind="stable", na_position="last")
    valid = (
        ordered_raw["text"].fillna("").astype(str).str.strip().ne("")
        & ordered_raw["created_at"].notna()
        & ~ordered_raw.duplicated("id", keep="first")
    )
    check(posts["id"].is_unique and set(posts["id"]) == set(ordered_raw.loc[valid, "id"]), "RQ2/RQ3 tidak memuat seluruh posting valid")
    check(set(topic["id"]).issubset(set(posts["id"])), "RQ1 bukan subset RQ2/RQ3")
    check(set(topic["duplicate_text_group"]).issubset(set(posts["duplicate_text_group"])), "RQ1 bukan subset RQ2/RQ3")
    missing_topic = posts["topic_document_id"].isna()
    check((~missing_topic | posts["text_embed"].fillna("").str.strip().eq("")).all(), "Posting dengan embedding tidak kosong kehilangan topic_document_id")
    mapped = posts.loc[~missing_topic]
    check(set(mapped["topic_document_id"]).issubset(set(topic["topic_document_id"])), "Pemetaan posting ke RQ1 tidak lengkap")
    check(mapped["topic_document_id"].eq(mapped["duplicate_text_group"]).all(), "Pemetaan posting tidak sesuai kelompok teks")
    check(topic["duplicate_text_group"].is_unique, "RQ1 masih memiliki exact duplicate")
    eligible = posts[posts["text_normalized"].fillna("").str.strip().ne("") & posts["text_embed"].fillna("").str.strip().ne("")]
    check(topic["id"].tolist() == eligible.drop_duplicates("duplicate_text_group", keep="first")["id"].tolist(), "Dokumen RQ1 bukan posting layak paling awal")
    results["rq_population_consistency"] = {"status": "passed"}

    required = {"id", "text", "text_minimal", "text_embed", "text_normalized", "text_topic", "topic_document_id", "dup_group_size", "source_keyword", "source_month", "window_capped", "uses_embege"}
    check(required.issubset(topic.columns), "Kolom wajib BERTopic tidak lengkap")
    check(topic["topic_document_id"].notna().all(), "topic_document_id kosong")
    results["bertopic_input_validation"] = {"status": "passed", "rows": len(topic)}

    check(all({"bio", "judge_score", "on_topic"}.isdisjoint(frame.columns) for frame in [posts, topic, audit]), "Kolom bio, judge_score, atau on_topic masih ada di output")
    results["output_privacy"] = {"status": "passed"}

    for name, frame in {"posts": posts, "topic": topic}.items():
        check(~frame["text_embed"].fillna("").str.contains("http", case=False).any(), f"URL masih ada di {name}.text_embed")
    check(topic["text_embed"].fillna("").str.strip().ne("").all(), "Embedding dokumen topik kosong")
    results["embedding_text"] = {"status": "passed"}

    group_sizes = posts.groupby("duplicate_text_group").size()
    for name, frame in {"posts": posts, "topic": topic}.items():
        check(frame["dup_group_size"].eq(frame["duplicate_text_group"].map(group_sizes)).all(), f"Ukuran kelompok {name} tidak cocok")
    results["duplicate_group_sizes"] = {"status": "passed"}

    check(~topic["text_topic"].fillna("").str.contains(r"\[url\]|\[user\]", case=False).any(), "Placeholder masih ada pada text_topic")
    results["topic_placeholders"] = {"status": "passed"}

    stopwords = (OUTPUT / "topic_stopwords.txt").read_text(encoding="utf-8").splitlines()
    check({"mbg", "embege", "url"}.issubset(stopwords) and not {"gizi", "program"}.intersection(stopwords), "Stopword topik tidak sesuai")
    check(stopwords == sorted(pipeline.STOPWORDS_ID | pipeline.QUERY_STOPWORDS | pipeline.FILLER_STOPWORDS), "Daftar stopword tidak lengkap/terurut")
    results["topic_stopwords"] = {"status": "passed"}

    for name, frame in {"posts": posts, "topic": topic}.items():
        check(pd.api.types.is_bool_dtype(frame["uses_embege"]) and frame["uses_embege"].notna().all(), f"{name}.uses_embege bukan boolean")
        check(frame["source_keyword"].notna().all(), f"{name}.source_keyword memiliki missing value")
        check(not frame["source_keyword"].str.contains(r" since|since_time", case=False, na=False).any(), f"{name}.source_keyword berisi query slot")
        check(frame["source_month"].str.fullmatch(r"\d{4}-\d{2}", na=False).all(), f"{name}.source_month tidak sesuai YYYY-MM")
        check(str(frame["window_capped"].dtype) == "boolean", f"{name}.window_capped bukan nullable boolean")
    check(topic[["source_month", "window_capped"]].reset_index(drop=True).equals(posts.set_index("id").loc[topic["id"], ["source_month", "window_capped"]].reset_index(drop=True)), "Metadata window dokumen topik berbeda dari posting sumber")
    results["source_metadata"] = {"status": "passed"}

    collection_log = ROOT / "collection_log.csv"
    if collection_log.exists():
        for name, frame in {"posts": posts, "topic": topic}.items():
            check(frame["window_capped"].notna().all(), f"{name}.window_capped memiliki missing value saat log tersedia")
        for row in pd.read_csv(collection_log, dtype="string", keep_default_na=False).itertuples(index=False):
            window = posts["source_keyword"].eq(row.keyword) & posts["source_month"].eq(row.month)
            check(posts.loc[window, "window_capped"].eq(row.capped == "True").all(), f"Flag capped tidak cocok untuk {row.keyword}, {row.month}")
    else:
        check(posts["window_capped"].isna().all() and topic["window_capped"].isna().all(), "Tanpa log, semua window_capped harus unknown")
    results["collection_window_mapping"] = {"status": "passed"}

    # split("\n") dan bukan splitlines(): teks boleh memuat U+2028 mentah
    records = [json.loads(line) for line in (OUTPUT / "posts_preprocessed.jsonl").read_text(encoding="utf-8").split("\n") if line]
    check(len(records) == len(posts) and list(records[0]) == list(posts.columns), "JSONL tidak sama dengan posts_rq2_rq3.csv (jumlah/kolom)")
    check([r["id"] for r in records] == posts["id"].tolist() and all(isinstance(r["id"], str) for r in records), "ID JSONL bukan string atau urutannya berbeda")
    check([r["text_embed"] for r in records] == posts["text_embed"].tolist(), "text_embed JSONL tidak sama dengan CSV")
    check(all(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", r["created_at"]) for r in records), "Format created_at JSONL bukan ISO 8601 UTC")
    check(all(r["window_capped"] in (True, False, None) and isinstance(r["uses_embege"], bool) for r in records), "Tipe boolean JSONL salah")
    results["jsonl_output"] = {"status": "passed", "rows": len(records)}

    check(all("pull_batch" in row for row in records), "Record JSONL kehilangan pull_batch")
    raw_batches = raw["pull_batch"].fillna("month_wide") if "pull_batch" in raw else pd.Series("month_wide", index=raw.index)
    batch_counts = Counter(raw_batches)
    for name, frame in {"posts": posts, "topic": topic, "jsonl": pd.DataFrame(records)}.items():
        check(frame["pull_batch"].isin(["month_wide", "slot_sample"]).all(), f"{name}.pull_batch tidak valid")
        if name != "topic":
            check(Counter(frame["pull_batch"]) == batch_counts, f"Hitungan batch {name} berbeda dari raw")
        slot = frame["pull_batch"].eq("slot_sample")
        check(frame.loc[slot, ["slot_since", "slot_until"]].notna().all().all(), f"{name}.slot_sample kehilangan batas slot")
        check(frame.loc[~slot, ["slot_since", "slot_until"]].isna().all().all(), f"{name}.month_wide memiliki batas slot")
        check(not frame["source_keyword"].str.contains(r" since|since_time", case=False, na=False).any(), f"{name}.source_keyword berisi query slot")
        check(frame.set_index("id")["pull_batch"].eq(raw_batches.set_axis(raw["id"]).reindex(frame["id"])).all(), f"{name}.pull_batch berbeda dari raw per ID")
    for batch in ["month_wide", "slot_sample"]:
        check(retention[f"pull_batch_{batch}_rows"] == batch_counts[batch], f"Retention {batch} berbeda dari raw")
    for column in ["pull_batch", "slot_since", "slot_until"]:
        check([row[column] for row in records] == posts[column].astype(object).where(posts[column].notna(), None).tolist(), f"JSONL.{column} berbeda dari CSV")
        check(topic[column].reset_index(drop=True).equals(posts.set_index("id").loc[topic["id"], column].reset_index(drop=True)), f"Metadata batch RQ1.{column} berbeda dari posting sumber")
    results["batch_provenance"] = {"status": "passed", "raw_counts": dict(batch_counts)}

    report = {
        "summary": {
            "automated_checks_passed": len(results),
            "automated_checks_failed": 0,
            "manual_semantic_reviews_completed": int(audit["review_status"].eq("selesai").sum()),
            "manual_semantic_reviews_total": len(audit),
        },
        "checks": results,
    }
    (OUTPUT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
