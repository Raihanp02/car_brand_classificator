"""Cached contract/session/index singletons, loaded once per process."""

from __future__ import annotations

import logging
from typing import Any

from app import config
from app.ml.contract import ModelContract, load_contract, verify_model_file
from app.ml.index import load_index
from app.ml.onnx_embedder import OnnxEmbedder

log = logging.getLogger("car_classifier")

_CONTRACT: dict[str, Any] = {}
_INDEX: dict[str, Any] = {}
_EMBEDDER: dict[str, Any] = {}


def reset_state() -> None:
    """Clear cached artifact state (tests only)."""
    _CONTRACT.clear()
    _INDEX.clear()
    _EMBEDDER.clear()


def get_contract() -> ModelContract:
    if _CONTRACT.get("ready"):
        return _CONTRACT["contract"]
    if _CONTRACT.get("error"):
        raise RuntimeError(_CONTRACT["error"])
    try:
        contract = load_contract(config.MODEL_META_PATH)
        _CONTRACT.update({"contract": contract, "ready": True})
        return contract
    except Exception as exc:  # fail fast with an explicit message
        _CONTRACT["error"] = str(exc)
        raise RuntimeError(str(exc)) from exc


def get_index() -> tuple[Any, dict]:
    """Return cached ``(faiss_index, meta)``, validated against the contract."""
    if _INDEX.get("ready"):
        return _INDEX["index"], _INDEX["meta"]
    if _INDEX.get("error"):
        raise RuntimeError(_INDEX["error"])
    try:
        contract = get_contract()
        index, meta = load_index(
            config.INDEX_PATH,
            expected_method=contract.method,
            expected_dim=contract.dim,
        )
        if index.ntotal == 0:
            raise ValueError("index is empty; rebuild it in the training repo")
        _INDEX.update({"index": index, "meta": meta, "ready": True})
        return index, meta
    except Exception as exc:  # fail fast with an explicit message
        _INDEX["error"] = str(exc)
        raise RuntimeError(str(exc)) from exc


def get_embedder() -> OnnxEmbedder:
    """Return the cached ONNX embedder (lazy: validates index before loading)."""
    if _EMBEDDER.get("ready"):
        return _EMBEDDER["embedder"]
    if _EMBEDDER.get("error"):
        raise RuntimeError(_EMBEDDER["error"])
    try:
        contract = get_contract()
        get_index()  # validates method/dim against the sidecar BEFORE the session loads
        verify_model_file(config.MODEL_PATH, contract)
        embedder = OnnxEmbedder(config.MODEL_PATH, contract)
        log.info(
            "onnx embedder loaded: method=%s dim=%d model=%s",
            contract.method,
            contract.dim,
            config.MODEL_PATH,
        )
        _EMBEDDER.update({"embedder": embedder, "ready": True})
        return embedder
    except Exception as exc:
        _EMBEDDER["error"] = str(exc)
        raise RuntimeError(str(exc)) from exc
