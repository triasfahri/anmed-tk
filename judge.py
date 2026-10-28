import re

import numpy as np


def load_encoder(name, max_seq=128, batch_size=64):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name)
    model.max_seq_length = max_seq

    def encode(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
        return np.asarray(
            model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )

    return encode


def mask_terms(text, terms):
    for term in sorted(terms, key=len, reverse=True):
        words = term.split()
        if not words:
            continue
        pattern = r"#?" + r"\s+".join(re.escape(word) for word in words)
        if len(words) > 1:
            pattern += "|#" + re.escape("".join(words))
        text = re.sub(r"(?<!\w)(?:" + pattern + r")(?!\w)", "", text, flags=re.IGNORECASE)
    return text


def prep(text, mask=()):
    text = mask_terms(text, mask).lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"#(?=\w)", "", text)
    return " ".join(text.split())


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not pos.size or not neg.size:
        raise ValueError("auc requires non-empty positive and negative scores")
    _, groups, counts = np.unique(
        np.concatenate((pos, neg)), return_inverse=True, return_counts=True
    )
    ranks = np.cumsum(counts) - (counts - 1) / 2
    u = ranks[groups[:pos.size]].sum() - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def youden_threshold(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not pos.size or not neg.size:
        raise ValueError("threshold requires non-empty positive and negative scores")
    thresholds = np.unique(np.concatenate((pos, neg)))
    tp = pos.size - np.searchsorted(np.sort(pos), thresholds, side="left")
    fp = neg.size - np.searchsorted(np.sort(neg), thresholds, side="left")
    # The common denominator keeps mathematical ties exact.
    j = tp * neg.size - fp * pos.size
    return float(thresholds[j == j.max()][-1])


def _cosine(centered, c):
    norms = np.linalg.norm(centered, axis=1) * np.linalg.norm(c)
    return np.divide(
        centered @ c, norms, out=np.zeros(len(centered), dtype=np.float32), where=norms > 0
    )


def score(texts, cal, encode, mask=()):
    prepared = [prep(text, mask) for text in texts]
    keep = [i for i, text in enumerate(prepared) if len(text.split()) >= 3]
    scores = np.full(len(texts), np.nan, dtype=np.float32)
    if keep:
        embeddings = np.asarray(encode([prepared[i] for i in keep]), dtype=np.float32)
        scores[keep] = _cosine(embeddings - cal["mu"], cal["c"])
    return scores


def calibrate(R, B, encode, mask):
    R = [prep(text, mask) for text in R]
    B = [prep(text, mask) for text in B]
    R = [text for text in R if len(text.split()) >= 3]
    B = [text for text in B if len(text.split()) >= 3]
    if not R or not B:
        raise ValueError("calibration needs texts with at least 3 tokens in both groups")
    embeddings = np.asarray(encode(R + B), dtype=np.float32)
    mu = embeddings.mean(axis=0)
    centered = embeddings - mu
    c = centered[:len(R)].mean(axis=0)
    norm = np.linalg.norm(c)
    if norm > 0:
        c = c / norm
    R_scores = _cosine(centered[:len(R)], c)
    B_scores = _cosine(centered[len(R):], c)
    quality = auc(R_scores, B_scores)
    if quality < 0.8:
        raise ValueError("judge does not separate round-1 posts from background; switch ENCODER")
    tau = youden_threshold(R_scores, B_scores)
    return {
        "mu": mu,
        "c": c,
        "tau": tau,
        "auc": quality,
        "b0": float(np.mean(B_scores >= tau)),
    }


def save_calibration(cal, path):
    np.savez(path, **cal)


def load_calibration(path):
    with np.load(path) as data:
        return {
            "mu": data["mu"],
            "c": data["c"],
            "tau": float(data["tau"]),
            "auc": float(data["auc"]),
            "b0": float(data["b0"]),
        }
