#!/usr/bin/env python3
"""Gabungkan koleksi month-wide dan slot sample tanpa ID ganda."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def find_source(filename: str) -> Path:
    for directory in [ROOT.parent / "data", ROOT.parent / "tugas-kelompok/data"]:
        path = directory / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"Sumber {filename} tidak ditemukan; gunakan --old/--new.")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def merge_raw(old: Path, new: Path, out: Path) -> None:
    backup = out.with_name("raw_data_MBG_batch1_month_wide.jsonl")
    if out.resolve() in {old.resolve(), new.resolve(), backup.resolve()}:
        raise ValueError("Output tidak boleh menimpa sumber atau arsip batch 1.")
    month_wide = read_jsonl(old)
    slot_sample = read_jsonl(new)
    rows = month_wide + slot_sample
    seen = set()
    for row in rows:
        post_id = str(row["id"])
        if post_id in seen:
            raise ValueError(f"ID ganda: {post_id}; tidak ada file yang ditulis.")
        seen.add(post_id)
    for row in month_wide:
        row.setdefault("pull_batch", "month_wide")
    content = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    if out.exists() and not backup.exists():
        out.rename(backup)
    out.write_text(content, encoding="utf-8")
    print(f"month_wide: {len(month_wide):,}")
    print(f"slot_sample: {len(slot_sample):,}")
    print(f"Total: {len(rows):,} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, help="Default: final_posts.jsonl pada folder data saudara.")
    parser.add_argument("--new", type=Path, help="Default: extra_posts.jsonl pada folder data saudara.")
    parser.add_argument("--out", type=Path, default=ROOT / "raw_data_MBG.jsonl")
    args = parser.parse_args()
    merge_raw(args.old or find_source("final_posts.jsonl"), args.new or find_source("extra_posts.jsonl"), args.out)


if __name__ == "__main__":
    main()
