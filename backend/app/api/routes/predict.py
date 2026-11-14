"""Prediction endpoints (single and batch share one batched path)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile, Depends
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from typing import Annotated
from dataclasses import asdict

from app.config import settings
from app.api import deps
from app.api.schemas import BatchPredictResponse
from app.api.deps import DecodedBatch, get_contract
from app.services.classifier import CarClassifier

log = logging.getLogger("car_classifier")

router = APIRouter()

# endpoint
@router.post(
    "/v1/predict/batch",
    response_model=BatchPredictResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(deps.authenticate)],   # runs auth; also sets request.state.key_id
)
async def predict_batch(
    request: Request,
    images: Annotated[DecodedBatch, Depends(deps.decoded_batch)],
    classifier: Annotated[CarClassifier, Depends(deps.get_classifier_service)],
):
    if images.pils:
        try:
            preds = await run_in_threadpool(classifier.predict, images.pils)
        except RuntimeError:
            log.exception("batch inference failed")
            raise HTTPException(500, "inference failed") from None

        for i, pred in zip(images.order, preds):
            images.results[i].update(asdict(pred))   # label, confidence, top_k

    first_ok = next((r for r in images.results if "label" in r), None)
    if first_ok:
        request.state.usage = {
            "label": first_ok["label"],
            "confidence": first_ok["confidence"],
        }

    return BatchPredictResponse(
        model_version=get_contract(request).model_version,
        results=images.results,
    )