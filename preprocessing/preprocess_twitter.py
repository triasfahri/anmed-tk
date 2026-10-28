#!/usr/bin/env python3
"""Pipeline preprocessing Twitter/X untuk BERTopic dan analisis deskriptif."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import unicodedata
from pathlib import Path

# Dependency dapat diinstal normal melalui pip atau disimpan lokal di .vendor.
ROOT = Path(__file__).resolve().parent
VENDOR_DIR = ROOT / ".vendor"
if VENDOR_DIR.is_dir():
    sys.path.insert(0, str(VENDOR_DIR))

import emoji
import pandas as pd

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")
HASHTAG_DETECT_RE = re.compile(r"#[A-Za-z0-9_]+")
SPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
REPEATED_CHAR_RE = re.compile(r"([A-Za-z])\1{2,}", re.I)
EMOJI_ALIAS_RE = re.compile(r":([a-zA-Z0-9_+\-]+):")
TOKEN_RE = re.compile(r"\[url\]|\[user\]|emoji_[a-z0-9_]+|[a-z0-9]+(?:-[a-z0-9]+)?", re.I)

SLANG_MAP = {
    "aj": "saja", "aja": "saja", "bgt": "banget", "bgitu": "begitu",
    "bener": "benar", "beneran": "benar-benar", "blm": "belum",
    "dgn": "dengan", "dpt": "dapat", "drpd": "daripada",
    "drpada": "daripada", "emang": "memang", "ga": "tidak", "gak": "tidak",
    "gimana": "bagaimana", "gmn": "bagaimana", "gk": "tidak", "kgak": "tidak",
    "kalo": "kalau", "kalopun": "kalaupun", "knp": "kenapa", "krn": "karena",
    "loe": "kamu", "mang": "memang", "mending": "lebih baik", "nggak": "tidak",
    "ngga": "tidak", "org": "orang", "sdh": "sudah", "slalu": "selalu",
    "tau": "tahu", "tdk": "tidak", "trs": "terus", "tuh": "itu",
    "udah": "sudah", "udh": "sudah", "yg": "yang",
}

# Negasi sengaja tidak dibuang.
STOPWORDS_ID = {
    "ada", "adalah", "agar", "akan", "aku", "anda", "apa", "atau", "bagai",
    "bagi", "bahwa", "banyak", "beberapa", "begitu", "bisa", "dalam", "dan",
    "dari", "dengan", "di", "dia", "ini", "itu", "jadi", "juga", "kami",
    "karena", "ke", "kita", "lebih", "maka", "mereka", "oleh", "pada", "para",
    "saat", "saja", "sangat", "saya", "sebagai", "seperti", "sudah", "supaya",
    "telah", "tentang", "tersebut", "untuk", "yang",
}
QUERY_STOPWORDS = {
    "mbg", "makan", "bergizi", "gratis", "sppg", "bgn", "embege", "badan",
    "nasional", "satuan", "pelayanan", "pemenuhan", "url", "user",
}
FILLER_STOPWORDS = {"ya", "sih", "kok", "lagi", "nya", "deh", "dong", "lah", "kan", "aja"}

HASHTAG_FIXES = {
    "beritaterkini": "berita terkini",
    "bicarafaktalewatberita": "bicara fakta lewat berita",
    "tniprima": "tni prima",
    "adadikompas": "ada di kompas",
    "cariberitaditvone": "cari berita di tvone",
    "infoterkini": "info terkini",
    "jagaindonesia": "jaga indonesia",
    "polripresisi": "polri presisi",
    "programpemerintah": "program pemerintah",
    "tniad": "tni ad",
    "awasimbg": "awasi mbg",
    "makanbergizigratis": "makan bergizi gratis",
    "badangizinasional": "badan gizi nasional",
    "programmbg": "program mbg",
    "lanjutkanmbg": "lanjutkan mbg",
    "keracunanmbg": "keracunan mbg",
    "dapurmbg": "dapur mbg",
    "kepalabgn": "kepala bgn",
    "polriuntukmasyarakat": "polri untuk masyarakat",
    "ditjenpkh": "ditjen pkh",
    "mbgpapua": "mbg papua",
}

REQUIRED_COLUMNS = {"id", "text", "created_at"}
NUMERIC_COLUMNS = ["likes", "retweets", "replies", "quotes", "followers"]


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(str(text)))
    return SPACE_RE.sub(" ", CONTROL_RE.sub(" ", text)).strip()


def segment_hashtag(tag: str) -> str:
    if tag.lower() in HASHTAG_FIXES:
        return HASHTAG_FIXES[tag.lower()]
    tag = tag.replace("_", " ")
    tag = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tag)
    tag = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", tag)
    tag = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", tag)
    return tag.lower()


def replace_social_elements(text: str) -> str:
    text = URL_RE.sub(" [URL] ", text)
    text = MENTION_RE.sub(" [USER] ", text)
    text = HASHTAG_RE.sub(lambda m: " " + segment_hashtag(m.group(1)) + " ", text)
    return SPACE_RE.sub(" ", text).strip()


def process_emoji(text: str) -> str:
    text = re.sub(r"[\U0001F3FB-\U0001F3FF]", "", text)
    demojized = emoji.demojize(text, language="en")
    return EMOJI_ALIAS_RE.sub(lambda m: " emoji_" + m.group(1).lower() + " ", demojized)


def has_emoji(text: str) -> bool:
    return process_emoji(text) != text


def normalize_repetitions(text: str) -> str:
    """Runtuhkan hanya repetisi 3+; massa dan Allah tetap utuh."""
    return REPEATED_CHAR_RE.sub(r"\1", text)


def normalize_slang(text: str) -> str:
    return re.sub(r"\b[A-Za-z]+\b", lambda m: SLANG_MAP.get(m.group(0).lower(), m.group(0)), text)


def clean_minimal(text: str) -> str:
    """Normalisasi minimal: konteks dan emoji asli dipertahankan."""
    return normalize_unicode(text)


def clean_embed(text: str) -> str:
    """Baseline embedding tanpa URL dan mention pembuka."""
    text = URL_RE.sub("", text)
    text = re.sub(r"^(?:@\w+\s*)+", "", text.lstrip())
    return SPACE_RE.sub(" ", text).strip()


def clean_normalized(text: str) -> str:
    """Kandidat embedding alternatif dengan slang dan emoji tekstual."""
    text = process_emoji(replace_social_elements(normalize_unicode(text)))
    return SPACE_RE.sub(" ", normalize_slang(normalize_repetitions(text.lower()))).strip()


def clean_topic(text: str) -> str:
    """Representasi c-TF-IDF/label topik, bukan input embedding utama."""
    tokens = [t.lower() for t in TOKEN_RE.findall(clean_normalized(text))]
    return " ".join(t for t in tokens if t not in STOPWORDS_ID and t not in {"[url]", "[user]"} and len(t) > 1)


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"id": "string", "parent_id": "string"})
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
        result = pd.DataFrame(records)
        for column in ["id", "parent_id"]:
            if column in result:
                result[column] = result[column].astype("string")
        return result
    try:
        return pd.read_json(path, lines=True, dtype={"id": "string", "parent_id": "string"})
    except ValueError:
        return pd.read_json(path, dtype={"id": "string", "parent_id": "string"})


def text_group_id(text: str) -> str:
    return "txt_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def map_window_capped(df: pd.DataFrame, collection_log: Path) -> pd.Series:
    if not collection_log.exists():
        print(f"Peringatan: {collection_log} tidak ditemukan; window_capped kosong.")
        return pd.Series(pd.NA, index=df.index, dtype="boolean")
    log = pd.read_csv(collection_log, dtype="string", keep_default_na=False)
    capped = log.set_index(["keyword", "month"])["capped"].str.strip().str.lower().map({
        "true": True, "1": True, "yes": True, "y": True,
        "false": False, "0": False, "no": False, "n": False,
    }).astype("boolean")
    return df[["source_keyword", "source_month"]].join(capped.rename("window_capped"), on=["source_keyword", "source_month"])["window_capped"]


def prepare_data(df: pd.DataFrame, collection_log: Path = ROOT / "collection_log.csv") -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")

    result = df.drop(columns=["bio", "judge_score", "on_topic"], errors="ignore").copy()
    result["id"] = result["id"].astype("string")
    if "parent_id" in result:
        result["parent_id"] = result["parent_id"].astype("string")
    result["text"] = result["text"].fillna("").astype(str)
    result["created_at"] = pd.to_datetime(result["created_at"], utc=True, errors="coerce")
    result = result.sort_values("created_at", kind="stable", na_position="last")
    for col in NUMERIC_COLUMNS:
        if col in result:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    result["is_missing_text"] = result["text"].str.strip().eq("")
    result["is_invalid_date"] = result["created_at"].isna()
    result["is_duplicate_id"] = result.duplicated("id", keep="first")
    result["text_minimal"] = result["text"].map(clean_minimal)
    duplicate_text = result["text_minimal"].str.replace(URL_RE, "", regex=True).str.replace(SPACE_RE, " ", regex=True).str.strip().str.lower()
    duplicate_text = duplicate_text.where(duplicate_text.ne(""), result["text_minimal"].str.lower())
    result["duplicate_text_group"] = duplicate_text.map(text_group_id)
    result["is_duplicate_text"] = result.duplicated("duplicate_text_group", keep="first")
    result["text_embed"] = result["text_minimal"].map(clean_embed)
    result["text_normalized"] = result["text"].map(clean_normalized)
    result["text_topic"] = result["text"].map(clean_topic)
    query = result["query"].astype("string") if "query" in result else pd.Series(pd.NA, index=result.index, dtype="string")
    parsed_source = {
        "source_keyword": query.str.split(" since:", n=1).str[0].str.strip().str.strip('"'),
        "source_month": query.str.extract(r"\bsince:(\d{4}-\d{2})-\d{2}", expand=False),
    }
    for column, fallback in parsed_source.items():
        result[column] = result[column].astype("string").fillna(fallback) if column in result else fallback
    result["pull_batch"] = result["pull_batch"].fillna("month_wide") if "pull_batch" in result else "month_wide"
    result["window_capped"] = map_window_capped(result, collection_log)
    result["uses_embege"] = result["text"].str.contains(r"(?<![A-Za-z0-9_])embege(?![A-Za-z0-9_])", case=False, na=False)
    result["char_count"] = result["text_minimal"].str.len()
    result["token_count_minimal"] = result["text_minimal"].str.split().str.len()
    result["token_count_normalized"] = result["text_normalized"].str.split().str.len()
    result["has_url"] = result["text"].str.contains(URL_RE, na=False)
    result["has_mention"] = result["text"].str.contains(MENTION_RE, na=False)
    result["has_hashtag"] = result["text"].str.contains(HASHTAG_DETECT_RE, na=False)
    result["has_emoji"] = result["text"].map(has_emoji)

    retention = {
        "input_rows": len(result),
        "missing_text_rows": int(result["is_missing_text"].sum()),
        "invalid_date_rows": int(result["is_invalid_date"].sum()),
        "duplicate_id_rows": int(result["is_duplicate_id"].sum()),
        "duplicate_text_rows": int(result["is_duplicate_text"].sum()),
    }
    return result, retention


def build_outputs(prepared: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    valid = ~prepared["is_missing_text"] & ~prepared["is_invalid_date"] & ~prepared["is_duplicate_id"]
    posts = prepared[valid].copy()
    posts["dup_group_size"] = posts.groupby("duplicate_text_group")["id"].transform("size")
    topic = posts.copy()
    topic = topic[topic["text_normalized"].str.strip().ne("") & topic["text_embed"].str.strip().ne("")].copy()
    topic = topic[~topic.duplicated("duplicate_text_group", keep="first")].copy()
    topic["topic_document_id"] = topic["duplicate_text_group"]

    representative = topic.set_index("duplicate_text_group")["topic_document_id"]
    posts["topic_document_id"] = posts["duplicate_text_group"].map(representative).astype("string")
    posts = posts.sort_values("created_at", kind="stable").reset_index(drop=True)
    topic = topic.sort_values("created_at", kind="stable").reset_index(drop=True)
    stages = {
        "all_valid_posts_rows": len(posts),
        "posts_rq2_rq3_rows": len(posts),
        "topic_rows_after_exact_text_deduplication": len(topic),
        "window_capped_true_rows": int(posts["window_capped"].eq(True).sum()),
        "window_capped_false_rows": int(posts["window_capped"].eq(False).sum()),
        "window_capped_unknown_rows": int(posts["window_capped"].isna().sum()),
        "pull_batch_month_wide_rows": int(posts["pull_batch"].eq("month_wide").sum()),
        "pull_batch_slot_sample_rows": int(posts["pull_batch"].eq("slot_sample").sum()),
        "population_policy": "all valid posts",
    }
    return posts, topic, stages


def make_audit_sample(posts: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    sampled = posts.sample(n=min(size, len(posts)), random_state=seed) if len(posts) else posts.head(0)
    columns = ["id", "text", "text_minimal", "text_embed", "text_normalized", "text_topic", "has_emoji", "has_hashtag"]
    audit = sampled[columns].copy()
    audit["meaning_preserved"] = ""
    audit["review_status"] = "belum_dinilai"
    audit["reviewer_notes"] = ""
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "raw_data_MBG.jsonl")
    parser.add_argument("--collection-log", type=Path, default=ROOT / "collection_log.csv")
    parser.add_argument("--posts-output", type=Path, default=ROOT / "output/posts_rq2_rq3.csv")
    parser.add_argument("--topic-output", type=Path, default=ROOT / "output/topic_documents_rq1.csv")
    parser.add_argument("--audit-output", type=Path, default=ROOT / "output/manual_quality_audit.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "output/preprocessing_report.json")
    parser.add_argument("--json-output", type=Path, default=ROOT / "output/posts_preprocessed.jsonl")
    parser.add_argument("--stopwords-output", type=Path, default=ROOT / "output/topic_stopwords.txt")
    parser.add_argument("--audit-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_jsonl(posts: pd.DataFrame, path: Path) -> None:
    """JSON Lines dengan kolom sama seperti CSV; NA menjadi null, waktu ISO 8601 UTC seperti data mentah."""
    records = posts.assign(created_at=posts["created_at"].map(lambda t: t.isoformat()))
    records = records.astype(object).where(records.notna(), None)
    with path.open("w", encoding="utf-8") as stream:
        for row in records.to_dict("records"):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.audit_size < 0:
        raise ValueError("--audit-size tidak boleh negatif.")
    prepared, retention = prepare_data(load_data(args.input), args.collection_log)
    posts, topic, stages = build_outputs(prepared)
    audit = make_audit_sample(posts, args.audit_size, args.seed)
    for path in [args.posts_output, args.topic_output, args.audit_output, args.report, args.stopwords_output, args.json_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    posts.to_csv(args.posts_output, index=False, encoding="utf-8")
    write_jsonl(posts, args.json_output)
    topic.to_csv(args.topic_output, index=False, encoding="utf-8")
    audit.to_csv(args.audit_output, index=False, encoding="utf-8")
    args.stopwords_output.write_text("\n".join(sorted(STOPWORDS_ID | QUERY_STOPWORDS | FILLER_STOPWORDS)) + "\n", encoding="utf-8")
    report = {
        "retention": retention | stages,
        "configuration": {
            "input": str(args.input),
            "population_policy": "all valid posts",
            "audit_size": args.audit_size, "audit_seed": args.seed,
            "emoji_backend": f"emoji {emoji.__version__}",
            "stopwords_output": str(args.stopwords_output),
            "collection_log": str(args.collection_log),
            "embedding_candidates": ["text_embed", "text_minimal", "text_normalized"],
            "topic_representation": "text_topic",
        },
        "date_min": posts["created_at"].min().isoformat() if len(posts) else None,
        "date_max": posts["created_at"].max().isoformat() if len(posts) else None,
        "max_token_count_minimal_whitespace": int(posts["token_count_minimal"].max()) if len(posts) else 0,
        "p95_token_count_minimal_whitespace": float(posts["token_count_minimal"].quantile(.95)) if len(posts) else 0,
        "note_token_length": "Audit whitespace awal; ulangi dengan tokenizer model embedding final.",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RQ2/RQ3: {len(posts):,} posting valid -> {args.posts_output}")
    print(f"RQ1: {len(topic):,} dokumen unik -> {args.topic_output}")
    print(f"Audit manual: {len(audit):,} sampel -> {args.audit_output}")
    print(f"Laporan -> {args.report}")
    print(f"Stopword topik -> {args.stopwords_output}")
    print(f"JSONL seluruh posting -> {args.json_output}")


if __name__ == "__main__":
    main()
