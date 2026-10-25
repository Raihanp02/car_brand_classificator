"""The train/serve contract emitted by the training repo.

``model.meta.json`` is the source of truth for the embedding method, dim, and
preprocessing. The backend refuses to load an index or session that disagrees
with it, so a stale artifact pair fails loudly instead of returning nonsense.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONTRACT_VERSION = 1


@dataclass(frozen=True)
class ModelContract:
    model_version: str
    method: str
    dim: int
    preprocessing: dict[str, Any] = field(default_factory=dict)
    backbone: str | None = None
    opset: int | None = None
    sha256: str | None = None
    contract_version: int = CONTRACT_VERSION


def load_contract(path: str | Path) -> ModelContract:
    """Read and validate ``model.meta.json``; raise ValueError when incomplete."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"model contract missing: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"model contract is not valid JSON: {p}: {exc}") from exc

    missing = [k for k in ("model_version", "method", "dim", "preprocessing") if k not in data]
    if missing:
        raise ValueError(f"model contract missing keys {missing}: {p}")
    dim = int(data["dim"])
    if dim <= 0:
        raise ValueError(f"model contract has invalid dim={dim}: {p}")
    preprocessing = data["preprocessing"]
    if not isinstance(preprocessing, dict):
        raise ValueError(f"model contract preprocessing must be an object: {p}")
    return ModelContract(
        model_version=str(data["model_version"]),
        method=str(data["method"]),
        dim=dim,
        preprocessing=preprocessing,
        backbone=data.get("backbone"),
        opset=data.get("opset"),
        sha256=data.get("sha256"),
        contract_version=int(data.get("contract_version", CONTRACT_VERSION)),
    )


def verify_model_file(path: str | Path, contract: ModelContract) -> None:
    """Check the ONNX file against the sha256 pinned in the contract."""
    if not contract.sha256:
        return
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != contract.sha256:
        raise ValueError(f"model sha256 mismatch: {path}")
