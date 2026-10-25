"""Prediction endpoints (single and batch share one batched path)."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile
from PIL import Image

from app import config
from app.api import deps
from app.auth.store import log_usage
from app.services import predictor, registry

log = logging.getLogger("car_classifier")

router = APIRouter()


def _logged(
    request: Request,
    key_id: int | None,
    endpoint: str,
    status_code: int,
    t0: float,
    label: str = "",
    confidence: float = 0.0,
) -> int:
    latency_ms = int((time.perf_counter() - t0) * 1000)
    try:
        log_usage(
            key_id,
            endpoint,
            label,
            confidence,
            latency_ms,
            status_code,
            deps.client_ip(request),
        )
    except Exception as exc:  # logging must never break the API
        log.warning("usage log failed: %s", exc)
    return latency_ms


@router.post("/v1/predict")
async def predict(
    request: Request,
    image: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    t0 = time.perf_counter()
    endpoint = "/v1/predict"
    try:
        key_id = deps.authenticate(request, x_api_key)
    except HTTPException as exc:
        _logged(request, None, endpoint, exc.status_code, t0)
        raise
    data = await image.read()
    name = image.filename or "image"
    if len(data) > config.MAX_BYTES:
        _logged(request, key_id, endpoint, 413, t0)
        raise HTTPException(status_code=413, detail=f"{name}: file too large (>8MB)")
    try:
        pil = predictor.decode_image(data, name)
    except ValueError as exc:
        _logged(request, key_id, endpoint, 400, t0)
        raise HTTPException(status_code=400, detail=str(exc)) from None
    try:
        (res,) = predictor.predict_pil([pil])  # single = batch of 1
    except RuntimeError as exc:
        _logged(request, key_id, endpoint, 500, t0)
        raise HTTPException(status_code=500, detail=str(exc)) from None
    latency_ms = _logged(
        request, key_id, endpoint, 200, t0, res["label"], res["confidence"]
    )
    prefix = (x_api_key or "")[:6]
    log.info(
        "predict label=%s conf=%.3f ms=%d key=%s",
        res["label"], res["confidence"], latency_ms, prefix,
    )
    contract = registry.get_contract()
    return {**res, "latency_ms": latency_ms, "model_version": contract.model_version}


@router.post("/v1/predict/batch")
async def predict_batch(
    request: Request,
    images: list[UploadFile] | None = File(default=None),
    x_api_key: str | None = Header(default=None),
):
    t0 = time.perf_counter()
    endpoint = "/v1/predict/batch"
    try:
        key_id = deps.authenticate(request, x_api_key)
    except HTTPException as exc:
        _logged(request, None, endpoint, exc.status_code, t0)
        raise
    if not images:
        _logged(request, key_id, endpoint, 400, t0)
        raise HTTPException(status_code=400, detail="empty batch: no images")
    if len(images) > config.MAX_FILES:
        _logged(request, key_id, endpoint, 413, t0)
        raise HTTPException(
            status_code=413, detail=f"max {config.MAX_FILES} images per batch"
        )
    pils: list[Image.Image] = []
    order: list[int] = []
    names = [f.filename or f"image_{i}" for i, f in enumerate(images)]
    results: list[dict] = [{} for _ in images]
    for i, f in enumerate(images):
        data = await f.read()
        if len(data) > config.MAX_BYTES:
            results[i] = {"filename": names[i], "error": "file too large (>8MB)"}
            continue
        try:
            pils.append(predictor.decode_image(data, names[i]))
            order.append(i)
        except ValueError as exc:
            results[i] = {"filename": names[i], "error": str(exc)}
    if pils:
        try:
            preds = predictor.predict_pil(pils)  # ONE embed_batch call for the batch
        except RuntimeError as exc:
            _logged(request, key_id, endpoint, 500, t0)
            raise HTTPException(status_code=500, detail=str(exc)) from None
        for i, pred in zip(order, preds):
            results[i] = {"filename": names[i], **pred}
    top1 = next((r.get("label", "") for r in results if "label" in r), "")
    conf = next((r.get("confidence", 0.0) for r in results if "label" in r), 0.0)
    _logged(request, key_id, endpoint, 200, t0, top1, conf)
    contract = registry.get_contract()
    return {"results": results, "model_version": contract.model_version}
