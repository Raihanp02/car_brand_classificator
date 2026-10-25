"""Image decode, ONNX embedding, FAISS search, and kNN voting."""

from __future__ import annotations

import io
from collections import Counter

from PIL import Image

from app import config
from app.ml.index import search_index
from app.services import registry


def decode_image(data: bytes, filename: str) -> Image.Image:
    """Pillow-verify raw bytes; raise ValueError (400-class) when unreadable."""
    if not data:
        raise ValueError(f"{filename}: empty image")
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise ValueError(f"{filename}: corrupt image") from None
    if max(img.size) > config.MAX_DIM:
        raise ValueError(f"{filename}: image dimensions too large")
    return img.convert("RGB")


def vote(hits: list[dict]) -> tuple[str, float, list[dict]]:
    """kNN majority vote (ties -> best rank) -> (label, confidence, top5).

    Hits lacking ``"class"`` (phantom FAISS rows, e.g. idx -1 when k>ntotal)
    are filtered before voting so an empty-string class can never win.
    """
    valid = [h for h in hits if h.get("class")]
    if not valid:
        raise RuntimeError("FAISS returned no valid neighbours; rebuild the index")
    top5 = [
        {"label": h.get("class", ""), "score": float(h.get("score", 0.0))}
        for h in valid
    ]
    counts = Counter(h.get("class", "") for h in valid)
    best_rank_score: dict[str, float] = {}
    for h in valid:  # hits arrive rank-sorted; first occurrence = best rank
        best_rank_score.setdefault(h.get("class", ""), float(h.get("score", 0.0)))
    label = max(counts, key=lambda c: (counts[c], best_rank_score[c]))
    return label, best_rank_score[label], top5


def predict_pil(images: list[Image.Image]) -> list[dict]:
    """One embed_batch call -> per-image {label, confidence, top5}."""
    embedder = registry.get_embedder()
    index, meta = registry.get_index()
    emb = embedder.embed_batch(images)
    batch_hits = search_index(emb, k=config.TOP_K, index=index, meta=meta)
    out = []
    for hits in batch_hits:
        label, confidence, top5 = vote(hits)
        out.append({"label": label, "confidence": confidence, "top5": top5})
    return out
