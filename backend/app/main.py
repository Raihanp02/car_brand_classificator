"""FastAPI application assembly and startup lifespan.

Runtime artifacts come from the training repo:
    artifacts/model.onnx, artifacts/model.meta.json,
    artifacts/index.faiss, artifacts/index_meta.json
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
import json
from time import time

from app import config
from app.api import deps
from app.api.routes import ROUTERS
from app.services import registry
from app.services.classifier import CarClassifier
from app.ml.contract import load_contract
from app.ml.index import ReferenceIndex
from app.ml.onnx_embedder import OnnxEmbedder
from app.middleware.usage import UsageMiddleware

log = logging.getLogger("car_classifier")


@asynccontextmanager
async def lifespan(app: FastAPI):
    contract = load_contract(config.MODEL_META_PATH)            # raises on failure
    embedder = OnnxEmbedder.load(config.EMBEDDER_PATH, contract)
    sidecar = json.loads(config.INDEX_META_PATH.read_text())
    index = ReferenceIndex.load(config.INDEX_PATH, sidecar["method"], sidecar["dim"])

    classifier = CarClassifier(embedder, index, top_k=config.TOP_K)
    
    app.state.contract = contract
    app.state.classifier_service = classifier

    log.info("startup: model=%s method=%s dim=%d",
             contract.model_version, contract.method, contract.dim)
    yield


app = FastAPI(title="car-classifier", lifespan=lifespan)
app.add_middleware(UsageMiddleware)
for router in ROUTERS:
    app.include_router(router)


def reset_state() -> None:
    """Clear cached artifacts and demo rate-limit state (tests only)."""
    registry.reset_state()
    deps.reset_demo_state()


__all__ = ["app", "config", "reset_state"]
