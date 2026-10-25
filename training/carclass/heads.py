"""Embedding-head registry (car-classifier v1).

Single source of truth for how an embedding is derived from DINOv2 hidden
states ``h: (B, 1+N, hidden)``. Both sides of the train/serve contract use
this module:

- training: ``carclass/model.py`` and ``carclass/embedder.py``
- serving: ``backend/app/ml/onnx_embedder.py`` (ONNX only, no torch)

``METHODS`` is derived from each head's ``dim`` so the dims cannot drift
from the implementations. ``ClsProj256Head`` subclasses ``nn.Sequential``
so trained state-dict keys stay ``proj.0.weight`` etc. across old and new
checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_METHOD = "cls_proj256"


class Head(nn.Module):
    dim: int

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: (B, 1+N, hidden)
        raise NotImplementedError


class ClsHead(Head):
    dim = 768

    def forward(self, h):
        return h[:, 0]


class PatchMeanHead(Head):
    dim = 768

    def forward(self, h):
        # NOTE: mean over h[:, 1:] includes register tokens — train and
        # serve must agree, and both go through this exact line.
        return h[:, 1:].mean(1)


class ClsPatchConcatHead(Head):
    dim = 1536

    def forward(self, h):
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)


class ClsProj256Head(nn.Sequential):
    """Subclasses Sequential so state-dict keys stay `proj.0.weight`, etc."""

    dim = 256

    def __init__(self):
        super().__init__(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 256))

    def forward(self, h):
        return super().forward(h[:, 0])


HEADS: dict[str, type[Head]] = {
    "cls_proj256": ClsProj256Head,
    "cls": ClsHead,
    "patch_mean": PatchMeanHead,
    "cls_patch_concat": ClsPatchConcatHead,
}
METHODS = {name: cls.dim for name, cls in HEADS.items()}  # derived, can't drift
