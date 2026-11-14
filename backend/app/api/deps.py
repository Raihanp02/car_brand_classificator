"""Authentication and per-client rate limiting for request handlers.

API keys are checked against SQLite (``app.auth.store``); keyless demo traffic
is limited by an in-memory sliding window per client IP.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from datetime import date
from dataclasses import dataclass
from PIL import Image, ImageOps, UnidentifiedImageError
import io 

from fastapi import HTTPException, Request
from typing import Annotated
from fastapi import Depends, File, HTTPException, UploadFile, Header
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.auth.store import key_daily_usage, key_rate_limit, validate_key
from app.ml.contract import ModelContract
from app.services.classifier import CarClassifier

# Keyless-demo sliding window: ip -> deque[timestamps].
_DEMO_HITS: dict[str, deque[float]] = defaultdict(deque)


def reset_demo_state() -> None:
    """Clear demo rate-limit state (tests only)."""
    _DEMO_HITS.clear()


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _demo_limit_per_day() -> int:
    try:
        return settings.DEMO_LIMIT_PER_TIME
    except ValueError:
        return 20


def check_demo_limit(request: Request) -> None:
    """Sliding window over the last 24h for keyless demo hits; 429 + Retry-After."""
    limit = _demo_limit_per_day()
    ip = client_ip(request)
    now = time.time()
    dq = _DEMO_HITS[ip]
    while dq and dq[0] <= now - settings.DEMO_WINDOW_S:
        dq.popleft()
    if len(dq) >= limit:
        retry_after = max(1, int(dq[0] + settings.DEMO_WINDOW_S - now) + 1)
        raise HTTPException(
            status_code=429,
            detail="demo rate limit exceeded (keyless); retry later or use an API key",
            headers={"Retry-After": str(retry_after)},
        )
    dq.append(now)


def check_key_limit(key_id: int) -> None:
    limit = key_rate_limit(key_id)
    used = key_daily_usage(key_id, date.today().isoformat())
    if used >= limit:
        raise HTTPException(
            status_code=429,
            detail="API key daily rate limit exceeded",
            headers={"Retry-After": "3600"},
        )


def authenticate(request: Request, x_api_key: str | None = Header(default=None)) -> int | None:
    """Return key_id for valid keys, None for keyless demo (rate-limited).

    Raises 401 for a present-but-invalid/revoked key, 429 when limits hit.
    """
    if x_api_key:
        key_id = validate_key(x_api_key)
        if key_id is None:
            raise HTTPException(status_code=401, detail="invalid or revoked API key")
        check_key_limit(key_id)
        return key_id
    check_demo_limit(request)
    return None


def get_classifier_service(request: Request) -> CarClassifier:
    return request.app.state.classifier_service


def get_contract(request: Request) -> ModelContract:
    return request.app.state.contract

# Image dependencies for request handlers
def valid_batch(
    images: list[UploadFile] = File(default=[]),
) -> list[UploadFile]:
    if not images:
        raise HTTPException(400, "empty batch: no images")
    if len(images) > settings.MAX_FILES:
        raise HTTPException(413, f"max {settings.MAX_FILES} images per batch")
    return images


@dataclass
class DecodedBatch:
    pils: list[Image.Image]      # successfully decoded images
    order: list[int]             # original index of each entry in pils
    results: list[dict]          # one slot per upload, errors prefilled

def decode_image(data: bytes, filename: str, max_dim: int) -> Image.Image:
    """Decode raw bytes to RGB; raise ValueError (400-class) when unusable."""
    if not data:
        raise ValueError(f"{filename}: empty image")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if max(img.size) > max_dim:          # header only, nothing decoded yet
                raise ValueError(f"{filename}: image dimensions too large")
            img.load()                           # full decode, raises if corrupt
            img = ImageOps.exif_transpose(img)   # respect phone orientation
            return img.convert("RGB")
    except ValueError:
        raise                                    # our own message above
    except Image.DecompressionBombError:
        raise ValueError(f"{filename}: image dimensions too large") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValueError(f"{filename}: corrupt image") from None

async def decoded_images(
    images: Annotated[list[UploadFile], Depends(valid_batch)],
) -> DecodedBatch:
    names = [f.filename or f"image_{i}" for i, f in enumerate(images)]
    results = [{"filename": n} for n in names]
    pils, order = [], []

    for i, f in enumerate(images):
        data = await f.read()
        if len(data) > settings.MAX_BYTES:
            results[i]["error"] = f"file too large (>{settings.MAX_BYTES // 2**20}MB)"
            continue
        try:
            img = await run_in_threadpool(decode_image, data, names[i])
        except ValueError as exc:
            results[i]["error"] = str(exc)
            continue
        pils.append(img)
        order.append(i)

    return DecodedBatch(pils, order, results)