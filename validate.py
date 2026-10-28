import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np

import collect
import judge


DATA_DIR = "data"


def sample(data_dir, n=100, seed=0):
    data_dir = Path(data_dir)
    path = data_dir / "validation_sample.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            if any(row["label"] for row in csv.DictReader(stream)):
                raise SystemExit(f"Refusing to overwrite labeled sample: {path}")
    with (data_dir / "final_posts.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    rows = [row for row in rows
            if judge.mask_terms(row["text"], collect.SEEDS) == row["text"]]
    chosen = random.Random(seed).sample(rows, min(n, len(rows)))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "text", "label"])
        writer.writeheader()
        writer.writerows({"id": row["id"], "text": row["text"], "label": ""}
                         for row in chosen)
    print(f"Novel stratum: {len(rows)} rows; sampled {len(chosen)}.")


def wilson(k, n, z=1.96):
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("wilson requires n > 0 and 0 <= k <= n")
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (0.0 if k == 0 else center - margin,
            1.0 if k == n else center + margin)


def score_labels(data_dir, encoder_name=None, encode=None):
    data_dir = Path(data_dir)
    with (data_dir / "validation_sample.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row["label"] not in {"", "0", "1"}:
            raise SystemExit(f"Invalid label {row['label']!r} for row id {row['id']}")
    labeled = [row for row in rows if row["label"] != ""]
    labels = np.array([row["label"] == "1" for row in labeled], dtype=bool)
    n, k = len(labeled), int(labels.sum())
    precision = k / n if n else float("nan")
    ci = wilson(k, n) if n else (float("nan"), float("nan"))
    encoder_name = collect.ENCODER if encoder_name is None else encoder_name
    if encode is None:
        encoder = None

        def encode(texts):
            nonlocal encoder
            if encoder is None:
                encoder = judge.load_encoder(encoder_name, collect.MAX_SEQ)
            return encoder(texts)

    recalibrated = encoder_name != collect.ENCODER
    if recalibrated:
        with (data_dir / "posts.jsonl").open(encoding="utf-8") as stream:
            posts = [json.loads(line) for line in stream if line.strip()]
        R = [row["text"] for row in posts
             if judge.mask_terms(row["text"], collect.SEEDS) != row["text"]]
        with (data_dir / "background.jsonl").open(encoding="utf-8") as stream:
            B = [json.loads(line)["text"] for line in stream if line.strip()]
        cal = judge.calibrate(R, B, encode, collect.SEEDS)
    else:
        cal = judge.load_calibration(data_dir / "calibration.npz")
    scores = judge.score([row["text"] for row in labeled], cal, encode, mask=collect.SEEDS)
    valid = ~np.isnan(scores)
    pos, neg = scores[valid & labels], scores[valid & ~labels]
    judge_auc = judge.auc(pos, neg) if pos.size and neg.size else float("nan")
    predicted = scores >= cal["tau"]
    tp, predicted_n = int((predicted & labels).sum()), int(predicted.sum())
    report = {
        "n_labeled": n,
        "n_skipped": len(rows) - n,
        "precision": precision,
        "ci": ci,
        "judge_auc": judge_auc,
        "judge_precision": tp / predicted_n if predicted_n else 0.0,
        "judge_recall": tp / k if k else 0.0,
        "n_nan": int((~valid).sum()),
        "tau": cal["tau"],
        "encoder": encoder_name,
    }
    print(f"Labeled: {n}; skipped (unsure): {report['n_skipped']}")
    print(f"Novel stratum precision: {precision:.3f} "
          f"(Wilson 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")
    print(f"Judge AUC: {judge_auc:.3f}")
    print(f"Judge precision/recall at tau={cal['tau']:.3f}: "
          f"{report['judge_precision']:.3f} / {report['judge_recall']:.3f}")
    print(f"NaN scores: {report['n_nan']} (treated as off-topic at tau)")
    print(f"Encoder: {encoder_name}")
    if recalibrated:
        print("Calibration: seed-matching rows in posts.jsonl approximate round-1 posts; "
              "background.jsonl supplies B.")
    if n < 200:
        print("AUC differences under ~0.05 are noise at n=100")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sample and audit human validation labels.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sample")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--encoder")
    args = parser.parse_args(argv)
    if args.command == "sample":
        sample(DATA_DIR)
    else:
        score_labels(DATA_DIR, args.encoder)


if __name__ == "__main__":
    main()
