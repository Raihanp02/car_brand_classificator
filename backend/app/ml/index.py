"""FAISS index loading and nearest-neighbour search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _meta_path_for(faiss_path: str | Path) -> Path:
    p = Path(faiss_path)
    if p.suffix == ".faiss":
        return p.with_name(p.stem + "_meta.json")
    return Path(str(p) + "_meta.json")


@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    meta: dict  # filepath, class, view, split, ...


class ReferenceIndex:
    def __init__(self, index, meta: dict):
        self._index = index
        self._entries: list[dict] = meta.get("entries", [])
        self.method: str = meta["method"]
        self.dim: int = meta["dim"]

    @classmethod
    def load(
        cls, faiss_path: str | Path, expected_method: str, expected_dim: int
    ) -> "ReferenceIndex":
        import faiss

        faiss_path = Path(faiss_path)
        meta_path = _meta_path_for(faiss_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"index sidecar missing: {meta_path}")
        meta = json.loads(meta_path.read_text())
        if meta.get("method") != expected_method or meta.get("dim") != expected_dim:
            raise ValueError(
                f"method/dim mismatch: index {meta.get('method')}/{meta.get('dim')}"
                f" vs expected {expected_method}/{expected_dim}"
            )
        return cls(faiss.read_index(str(faiss_path)), meta)

    def search(self, emb: np.ndarray, k: int = 5) -> list[list[Hit]]:
        """Batched search. emb is (B, dim) or (dim,); always returns one list per query."""
        q = np.asarray(emb, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        scores, ids = self._index.search(q, k)

        out: list[list[Hit]] = []
        for qi in range(q.shape[0]):
            hits = []
            for rank, (idx, s) in enumerate(zip(ids[qi], scores[qi]), start=1):
                if not 0 <= idx < len(self._entries):  # FAISS phantom hit (-1)
                    continue
                hits.append(Hit(rank, float(s), self._entries[idx]))
            out.append(hits)
        return out

    def search_one(self, emb: np.ndarray, k: int = 5) -> list[Hit]:
        return self.search(emb, k)[0]