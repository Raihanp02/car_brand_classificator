"""Unified DINOv2 embedder + FAISS index helpers (car-classifier v1).

Backbone (exact): ``facebook/dinov2-with-registers-base``.
Default method: ``cls_proj256`` (dim 256, L2-normalized).

Methods:
    cls               CLS token, dim 768
    patch_mean        mean of patch tokens, dim 768
    cls_patch_concat  concat(CLS, patch_mean), dim 1536
    cls_proj256       MLP-projected CLS token, dim 256 (default)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from transformers import AutoModel

from app.heads import DEFAULT_METHOD, HEADS, METHODS

BACKBONE = "facebook/dinov2-with-registers-base"
# Re-exported for api.py / eval.py / build_index.py (canonical home: app.heads).

DEFAULT_INDEX = Path("models/index.faiss")
DEFAULT_META = Path("models/index_meta.json")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class CarEmbedder(nn.Module):
    def __init__(
        self,
        backbone: str = BACKBONE,
        method: str = DEFAULT_METHOD,
        checkpoint: str | Path | None = None,
    ):
        super().__init__()
        if method not in HEADS:
            raise ValueError(f"unknown method {method!r}; expected one of {sorted(HEADS)}")
        self.method = method
        self.backbone = AutoModel.from_pretrained(backbone)
        self.proj = HEADS[method]()      # name kept as `proj` for checkpoint compat
        self.dim = self.proj.dim
        self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                             T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
        if checkpoint:
            self.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True),
                strict=False,
            )
        self.eval()

    @torch.no_grad()
    def embed_batch(self, images: list[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.tf(im.convert("RGB")) for im in images])
        h = self.backbone(pixel_values=x).last_hidden_state
        out = self.proj(h)
        return out / out.norm(dim=-1, keepdim=True)


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
        emb: single (dim,) or batched (B, dim) query embedding(s),
            torch.Tensor or numpy array. Must be L2-normalized (as produced
            by :meth:`CarEmbedder.embed_batch`); IndexFlatIP then ranks by
            cosine similarity.
        k: number of neighbours to return.
        index: pre-loaded FAISS index; loads ``models/index.faiss`` if None.
        meta: sidecar meta dict for ``index``; loads ``models/index_meta.json``
            if None.

    Returns:
        For a single query: a list of the top-k ``{rank, filepath, class,
        view, split, score}`` dicts. For a batch: a list per query.
    """
    import numpy as np

    if index is None or meta is None:
        # Default: make the plain `search_index(emb, k=5)` form work against
        # the on-disk default index via its sidecar meta.
        if not DEFAULT_META.exists():
            raise FileNotFoundError(
                f"no index passed and default sidecar {DEFAULT_META} missing;"
                " build one with backend/scripts/build_index.py"
            )
        sidecar = json.loads(DEFAULT_META.read_text())
        index, meta = load_index(
            DEFAULT_INDEX,
            expected_method=sidecar["method"],
            expected_dim=sidecar["dim"],
        )
    entries = meta.get("entries", [])

    if isinstance(emb, torch.Tensor):
        q = emb.detach().cpu().to(torch.float32).numpy()
    else:
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
