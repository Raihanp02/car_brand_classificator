# app/services/classifier.py
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from PIL import Image

from app.error import InferenceError
from app.ml.index import Hit, ReferenceIndex


@dataclass(frozen=True)
class Neighbour:
    label: str
    score: float


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    top5: list[Neighbour]


def vote(hits: list[Hit]) -> Prediction:
    """kNN majority vote (ties -> best rank). Pure function, easy to unit test."""
    valid = [h for h in hits if h.meta.get("class")]
    if not valid:
        raise InferenceError("no valid neighbours; rebuild the index")

    counts = Counter(h.meta["class"] for h in valid)
    best_score: dict[str, float] = {}
    for h in valid:  # hits arrive rank-sorted; first occurrence = best rank
        best_score.setdefault(h.meta["class"], h.score)

    label = max(counts, key=lambda c: (counts[c], best_score[c]))
    top5 = [Neighbour(h.meta["class"], h.score) for h in valid]
    return Prediction(label, best_score[label], top5)


class CarClassifier:
    def __init__(self, embedder, index: ReferenceIndex, top_k: int):
        self._embedder = embedder
        self._index = index
        self._top_k = top_k

    def predict(self, images: list[Image.Image]) -> list[Prediction]:
        """One embed_batch call -> one Prediction per image."""
        emb = self._embedder.embed_batch(images)
        batch_hits = self._index.search(emb, k=self._top_k)
        return [vote(hits) for hits in batch_hits]