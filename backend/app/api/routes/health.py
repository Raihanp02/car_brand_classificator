"""Health and class-inventory endpoints."""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, HTTPException

from app.services import registry

router = APIRouter()


@router.get("/health")
def health():
    try:
        contract = registry.get_contract()
        index, meta = registry.get_index()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    return {
        "status": "ok",
        "model_version": contract.model_version,
        "index_size": int(index.ntotal),
        "dim": int(meta.get("dim", 0)),
    }


@router.get("/v1/classes")
def classes():
    try:
        _, meta = registry.get_index()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    counts = Counter(e.get("class", "") for e in meta.get("entries", []))
    return {
        "classes": [{"name": n, "count": c} for n, c in sorted(counts.items())],
        "total": len(counts),
    }
