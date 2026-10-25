"""FastAPI application assembly and startup lifespan.

Runtime artifacts come from the training repo:
    artifacts/model.onnx, artifacts/model.meta.json,
    artifacts/index.faiss, artifacts/index_meta.json
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import config
from app.api import deps
from app.api.routes import ROUTERS
from app.services import registry

log = logging.getLogger("car_classifier")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        contract = registry.get_contract()
    except RuntimeError as exc:
        log.warning("model contract unavailable (endpoints will 500): %s", exc)
    else:
        log.info(
            "startup contract: model=%s method=%s dim=%d",
            contract.model_version,
            contract.method,
            contract.dim,
        )
    try:  # preload index ONCE at startup; endpoints 500 if this failed
        registry.get_index()
    except RuntimeError as exc:
        log.warning("index preload failed (endpoints will 500): %s", exc)
    yield


app = FastAPI(title="car-classifier", lifespan=lifespan)
for router in ROUTERS:
    app.include_router(router)


def reset_state() -> None:
    """Clear cached artifacts and demo rate-limit state (tests only)."""
    registry.reset_state()
    deps.reset_demo_state()


__all__ = ["app", "config", "reset_state"]
