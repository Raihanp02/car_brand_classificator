"""FastAPI backend (car-classifier v1).

Routes: ``GET /health``, ``POST /v1/predict``, ``POST /v1/predict/batch``,
``GET /v1/classes``, ``POST /admin/keys``. OpenAPI at ``/docs``.

Binding decisions honored:
(a) ``models/`` resolves from the repo root, never cwd-relative; the FAISS
    index + meta load ONCE (cached singletons) and are reused per request.
(b) Real sha256 ``key_hash`` compare with ``revoked`` check (``auth``) and a
    ``usage_logs`` insert per predict call (success and failure).
(c) ``POST /admin/keys`` is operator-only via the ``ADMIN_TOKEN`` env var
    (``X-Admin-Token`` header must match; else 403).

Single predict goes through the same batched path (batch of 1); batch uses
ONE ``embed_batch`` call with per-item ``{error}`` entries that never fail
the whole batch.
"""

from __future__ import annotations

import io
import logging
import os
import time
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel

from app.auth import create_key, key_daily_usage, key_rate_limit, log_usage, validate_key
from app.embedder import DEFAULT_METHOD, METHODS, search_index

log = logging.getLogger("car_classifier")

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "models" / "index.faiss"
META_PATH = REPO_ROOT / "models" / "index_meta.json"
CHECKPOINT_PATH = REPO_ROOT / "models" / "dino_triplet.pt"

MODEL_VERSION = "registers-base_triplet_v1"
MAX_FILES = 10
MAX_BYTES = 8 * 1024 * 1024  # 8MB per image
MAX_DIM = 8000  # max width/height in px
TOP_K = 5
DEMO_WINDOW_S = 24 * 3600  # keyless demo sliding window

# Cached singletons: loaded once, reused per request (never reloaded per call).
_INDEX: dict[str, Any] = {}
_EMBEDDER: dict[str, Any] = {}
# Keyless-demo sliding window: ip -> deque[timestamps].
_DEMO_HITS: dict[str, deque[float]] = defaultdict(deque)


def reset_state() -> None:
    """Clear cached index/embedder/rate-limit state (tests only)."""
    _INDEX.clear()
    _EMBEDDER.clear()
    _DEMO_HITS.clear()


def _demo_limit_per_day() -> int:
    try:
        return int(os.environ.get("DEMO_LIMIT_PER_DAY", "20"))
    except ValueError:
        return 20


def _resolve_embed_cfg() -> tuple[str, int, Path | None]:
    """Single MODEL/CHECKPOINT resolution (embedder side, env override allowed).

    Method from ``MODEL`` (else ``CAR_METHOD``, else ``cls_proj256``); dim from
    ``METHODS[method]`` (same pattern as ``eval.py --method``). Checkpoint from
    ``CHECKPOINT`` env, else ``models/dino_triplet.pt`` when present, else None
    (clean absence keeps the random-projection baseline behavior).
    """
    method = os.environ.get("MODEL", os.environ.get("CAR_METHOD", DEFAULT_METHOD))
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(METHODS)}")
    dim = METHODS[method]
    ckpt_env = os.environ.get("CHECKPOINT")
    if ckpt_env:
        checkpoint: Path | None = Path(ckpt_env)
    else:
        checkpoint = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None
    return method, dim, checkpoint


def get_index() -> tuple[Any, dict]:
    """Return cached ``(faiss_index, meta)``, loading once from repo-root models/.

    Expected method/dim come from the embedder-side resolution
    (:func:`_resolve_embed_cfg`), never from the sidecar under test, so a
    method/dim mismatch refuses instead of comparing the index to itself.
    """
    if _INDEX.get("ready"):
        return _INDEX["index"], _INDEX["meta"]
    if _INDEX.get("error"):
        raise RuntimeError(_INDEX["error"])
    try:
        from app.embedder import load_index

        expected_method, expected_dim, _ = _resolve_embed_cfg()
        index, meta = load_index(
            INDEX_PATH,
            expected_method=expected_method,
            expected_dim=expected_dim,
        )
        if index.ntotal == 0:
            raise ValueError("index is empty; run backend/scripts/build_index.py")
        _INDEX.update({"index": index, "meta": meta, "ready": True})
        return index, meta
    except Exception as exc:  # fail fast with an explicit message
        _INDEX["error"] = str(exc)
        raise RuntimeError(str(exc)) from exc


def get_embedder() -> Any:
    """Return the cached ``CarEmbedder`` (lazy: loads trained weights when present)."""
    if _EMBEDDER.get("ready"):
        return _EMBEDDER["embedder"]
    from app.embedder import CarEmbedder

    method, dim, checkpoint = _resolve_embed_cfg()
    get_index()  # validates method/dim against the sidecar BEFORE weights load
    embedder = CarEmbedder(method=method, checkpoint=checkpoint)
    log.info("embedder loaded: method=%s dim=%d checkpoint=%s", method, dim, checkpoint)
    _EMBEDDER.update({"embedder": embedder, "ready": True})
    return embedder


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_demo_limit(request: Request) -> None:
    """Sliding window over the last 24h for keyless demo hits; 429 + Retry-After."""
    limit = _demo_limit_per_day()
    ip = _client_ip(request)
    now = time.time()
    dq = _DEMO_HITS[ip]
    while dq and dq[0] <= now - DEMO_WINDOW_S:
        dq.popleft()
    if len(dq) >= limit:
        retry_after = max(1, int(dq[0] + DEMO_WINDOW_S - now) + 1)
        raise HTTPException(
            status_code=429,
            detail="demo rate limit exceeded (keyless); retry later or use an API key",
            headers={"Retry-After": str(retry_after)},
        )
    dq.append(now)


def _check_key_limit(key_id: int) -> None:
    from datetime import date

    limit = key_rate_limit(key_id)
    used = key_daily_usage(key_id, date.today().isoformat())
    if used >= limit:
        raise HTTPException(
            status_code=429,
            detail="API key daily rate limit exceeded",
            headers={"Retry-After": "3600"},
        )


def _authenticate(request: Request, x_api_key: str | None) -> int | None:
    """Return key_id for valid keys, None for keyless demo (rate-limited).

    Raises 401 for a present-but-invalid/revoked key, 429 when limits hit.
    """
    if x_api_key:
        key_id = validate_key(x_api_key)
        if key_id is None:
            raise HTTPException(status_code=401, detail="invalid or revoked API key")
        _check_key_limit(key_id)
        return key_id
    _check_demo_limit(request)
    return None


def _decode_image(data: bytes, filename: str) -> Image.Image:
    """Pillow-verify raw bytes; raise ValueError (400-class) when unreadable."""
    if not data:
        raise ValueError(f"{filename}: empty image")
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise ValueError(f"{filename}: corrupt image") from None
    if max(img.size) > MAX_DIM:
        raise ValueError(f"{filename}: image dimensions too large")
    return img.convert("RGB")


def _vote(hits: list[dict]) -> tuple[str, float, list[dict]]:
    """kNN majority vote (ties -> best rank) -> (label, confidence, top5).

    Hits lacking ``"class"`` (phantom FAISS rows, e.g. idx -1 when k>ntotal)
    are filtered before voting so an empty-string class can never win.
    """
    valid = [h for h in hits if h.get("class")]
    if not valid:
        raise RuntimeError("FAISS returned no valid neighbours; rebuild the index")
    top5 = [
        {"label": h.get("class", ""), "score": float(h.get("score", 0.0))}
        for h in valid
    ]
    counts = Counter(h.get("class", "") for h in valid)
    best_rank_score: dict[str, float] = {}
    for h in valid:  # hits arrive rank-sorted; first occurrence = best rank
        best_rank_score.setdefault(h.get("class", ""), float(h.get("score", 0.0)))
    label = max(counts, key=lambda c: (counts[c], best_rank_score[c]))
    return label, best_rank_score[label], top5


def _predict_pil(images: list[Image.Image]) -> list[dict]:
    """One embed_batch call -> per-image {label, confidence, top5} (no latency/version)."""
    import torch

    embedder = get_embedder()
    index, meta = get_index()
    with torch.no_grad():
        emb = embedder.embed_batch(images)
    batch_hits = search_index(emb, k=TOP_K, index=index, meta=meta)
    out = []
    for hits in batch_hits:
        label, confidence, top5 = _vote(hits)
        out.append({"label": label, "confidence": confidence, "top5": top5})
    return out


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
        log_usage(key_id, endpoint, label, confidence, latency_ms, status_code, _client_ip(request))
    except Exception as exc:  # logging must never break the API
        log.warning("usage log failed: %s", exc)
    return latency_ms


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        method, dim, checkpoint = _resolve_embed_cfg()
    except ValueError as exc:
        log.warning("embed config invalid (endpoints will 500): %s", exc)
    else:
        log.info(
            "startup embed config: method=%s dim=%d checkpoint=%s",
            method, dim, checkpoint,
        )
    try:  # preload index + meta ONCE at startup; endpoints 500 if this failed
        get_index()
    except RuntimeError as exc:
        log.warning("index preload failed (endpoints will 500): %s", exc)
    yield


app = FastAPI(title="car-classifier", lifespan=lifespan)


@app.get("/health")
def health():
    try:
        index, meta = get_index()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "index_size": int(index.ntotal),
        "dim": int(meta.get("dim", 0)),
    }


@app.get("/v1/classes")
def classes():
    try:
        _, meta = get_index()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    counts = Counter(e.get("class", "") for e in meta.get("entries", []))
    return {
        "classes": [{"name": n, "count": c} for n, c in sorted(counts.items())],
        "total": len(counts),
    }


@app.post("/v1/predict")
async def predict(
    request: Request,
    image: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    t0 = time.perf_counter()
    endpoint = "/v1/predict"
    try:
        key_id = _authenticate(request, x_api_key)
    except HTTPException as exc:
        _logged(request, None, endpoint, exc.status_code, t0)
        raise
    data = await image.read()
    name = image.filename or "image"
    if len(data) > MAX_BYTES:
        _logged(request, key_id, endpoint, 413, t0)
        raise HTTPException(status_code=413, detail=f"{name}: file too large (>8MB)")
    try:
        pil = _decode_image(data, name)
    except ValueError as exc:
        _logged(request, key_id, endpoint, 400, t0)
        raise HTTPException(status_code=400, detail=str(exc)) from None
    try:
        (res,) = _predict_pil([pil])  # single = batch of 1, identical path
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
    return {**res, "latency_ms": latency_ms, "model_version": MODEL_VERSION}


@app.post("/v1/predict/batch")
async def predict_batch(
    request: Request,
    images: list[UploadFile] | None = File(default=None),
    x_api_key: str | None = Header(default=None),
):
    t0 = time.perf_counter()
    endpoint = "/v1/predict/batch"
    try:
        key_id = _authenticate(request, x_api_key)
    except HTTPException as exc:
        _logged(request, None, endpoint, exc.status_code, t0)
        raise
    if not images:
        _logged(request, key_id, endpoint, 400, t0)
        raise HTTPException(status_code=400, detail="empty batch: no images")
    if len(images) > MAX_FILES:
        _logged(request, key_id, endpoint, 413, t0)
        raise HTTPException(
            status_code=413, detail=f"max {MAX_FILES} images per batch"
        )
    pils: list[Image.Image] = []
    order: list[int] = []
    names = [f.filename or f"image_{i}" for i, f in enumerate(images)]
    results: list[dict] = [{} for _ in images]
    for i, f in enumerate(images):
        data = await f.read()
        if len(data) > MAX_BYTES:
            results[i] = {"filename": names[i], "error": "file too large (>8MB)"}
            continue
        try:
            pils.append(_decode_image(data, names[i]))
            order.append(i)
        except ValueError as exc:
            results[i] = {"filename": names[i], "error": str(exc)}
    if pils:
        try:
            preds = _predict_pil(pils)  # ONE embed_batch call for the batch
        except RuntimeError as exc:
            _logged(request, key_id, endpoint, 500, t0)
            raise HTTPException(status_code=500, detail=str(exc)) from None
        for i, pred in zip(order, preds):
            results[i] = {"filename": names[i], **pred}
    top1 = next((r.get("label", "") for r in results if "label" in r), "")
    conf = next((r.get("confidence", 0.0) for r in results if "label" in r), 0.0)
    _logged(request, key_id, endpoint, 200, t0, top1, conf)
    return {"results": results, "model_version": MODEL_VERSION}


class KeyRequest(BaseModel):
    name: str = "default"


@app.post("/admin/keys")
def admin_create_key(payload: KeyRequest, x_admin_token: str | None = Header(default=None)):
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=403, detail="forbidden: operator only")
    raw = create_key(payload.name)
    return {"name": payload.name, "api_key": raw}
