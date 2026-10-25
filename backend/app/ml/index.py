"""FAISS index loading and nearest-neighbour search."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app import config


def _meta_path_for(faiss_path: str | Path) -> Path:
    p = Path(str(faiss_path))
    if p.suffix == ".faiss":
        return p.with_name(p.stem + "_meta.json")
    return Path(str(faiss_path) + "_meta.json")


def load_index(faiss_path: str | Path, expected_method: str, expected_dim: int):
    """Load a FAISS index, refusing method/dim mismatches.

    Raises:
        ValueError: if the sidecar meta is missing or its method/dim
            does not match ``expected_method``/``expected_dim``.
    """
    import faiss

    faiss_path = Path(faiss_path)
    meta_path = _meta_path_for(faiss_path)
    meta = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    if meta is None:
        meta = {"method": None, "dim": None}
    if meta.get("method") != expected_method or meta.get("dim") != expected_dim:
        raise ValueError(
            f"method/dim mismatch: index {meta.get('method')}/{meta.get('dim')}"
            f" vs expected {expected_method}/{expected_dim}"
        )
    return faiss.read_index(str(faiss_path)), meta


def search_index(emb, k: int = 5, index=None, meta: dict | None = None) -> list:
    """Nearest-neighbour search over a loaded (or default on-disk) index.

    Args:
        emb: single (dim,) or batched (B, dim) L2-normalized query embedding(s).
        k: number of neighbours to return.
        index: pre-loaded FAISS index; loads the configured index if None.
        meta: sidecar meta dict for ``index``; loads the configured meta if None.

    Returns:
        For a single query: a list of the top-k ``{rank, filepath, class,
        view, split, score}`` dicts. For a batch: a list per query.
    """
    if index is None or meta is None:
        if not config.INDEX_META_PATH.exists():
            raise FileNotFoundError(
                f"no index passed and default sidecar {config.INDEX_META_PATH} missing;"
                " publish artifacts from the training repo"
            )
        sidecar = json.loads(config.INDEX_META_PATH.read_text())
        index, meta = load_index(
            config.INDEX_PATH,
            expected_method=sidecar["method"],
            expected_dim=sidecar["dim"],
        )
    entries = meta.get("entries", [])

    q = np.asarray(emb, dtype=np.float32)
    single = q.ndim == 1
    if single:
        q = q[None, :]
    scores, ids = index.search(q, k)
    out = []
    for qi in range(q.shape[0]):
        hits = []
        for rank, (idx, s) in enumerate(zip(ids[qi], scores[qi])):
            # Skip phantom hits: FAISS returns idx=-1 (and no entry) when
            # k > ntotal. They carry no "class" and must never be votable.
            if not 0 <= idx < len(entries):
                continue
            e = dict(entries[idx])
            e.update({"rank": rank + 1, "score": float(s)})
            hits.append(e)
        out.append(hits)
    return out[0] if single else out
