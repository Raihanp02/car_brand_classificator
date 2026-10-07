# Car Classifier API + Demo — Design Spec

Date: 2026-10-07
Status: approved in chat (awaiting file review)
Scope: v1 — triplet DINOv2 open-set classifier, FastAPI backend, Next.js demo, demo + manual API keys (no Stripe).

## 1. Intent (agreed)

- Build subscribable car-classifier: API service + website demo to try before subscribing.
- Output: brand + type, e.g. `toyota_avanza`. Year/variant not required in v1 but schema allows finer granularity later.
- Model: DINOv2 similarity on owned dataset, open-set add (new brand = add images + reindex, no retrain).
- Batch: model `embed_batch()` accepts batched tensors efficiently; API accepts single or up to N images in one call via same batched path.
- Split frontend/backend. Frontend: Next.js. Backend: Python FastAPI.
- Monetization v1: rate-limited demo + manual API keys. Full Stripe/accounts deferred.
- Dataset reality (2026-10-07): `car_brands/` 69 classes, 829 images, imbalanced 1–49, mostly folder-per-class with inconsistent `front/back/side/` subfolders.

Success criteria v1:
- top-1 >= 80%, top-5 >= 95% on held-out (reporting same-view and cross-view separately).
- < 1s CPU per image end-to-end (embed + kNN), batched faster per image than singles.
- Demo converts: visitor can upload 1–10 images and see top-1 + confidence + top-5 without signup.

## 2. Architecture

```
car_brands/<class>/<view>/*.jpg
training/                        # existing triplet trainer (fixed, kept)
backend/
  app/
    embedder.py                  # unified loader: backbone + method strategy
    api.py                       # FastAPI: /v1/predict, /v1/predict/batch, /v1/classes, /health
    auth.py                      # X-API-Key + IP rate limit
  scripts/
    build_index.py               # embed all refs -> index.faiss + index_meta.json (replaces ViT/add_image.py)
    eval.py                      # same-view / cross-view top-k
    clean_dataset.py             # flatten views -> metadata.csv, dedupe, verify
  requirements.txt               # fixed (torch, transformers, faiss/annoy, fastapi, etc.)
tools/labeling/                  # moved from ViT: naming_pipeline, prompt_mapping, brand_classifier
web/                             # Next.js demo (new): upload button + dropzone, 1–10 images, result cards
models/                           # gitignored: *.pt, index.faiss, index_meta.json
docs/superpowers/specs/          # this file
```

Backend owns ML, keys, limits. Web is stateless, calls backend via `NEXT_PUBLIC_API_URL`. Contract: OpenAPI JSON. Deploy independently (backend VPS/HF Space CPU, web Vercel). SQLite for keys + usage logs in v1 (Postgres later).

## 3. Model core (locked)

- Backbone: `facebook/dinov2-with-registers-base` (registers reduce artifact tokens; approved over plain base).
- Fine-tune: triplet with view-aware curriculum. Freeze all but last 2 blocks (or LoRA r=16). PKSampler P×K guarantees positives. Phase 1 positives = same class + same view (easy, retains appearance similarity). Phase 2 mix in same class + different view as hard positives (learns angle invariance). Batch-hard mining, margin 0.2, ImageNet norm, 224px, label-preserving aug (no aggressive crops that remove badges/grilles; color jitter + CCTV compression aug allowed).
- 1-shot classes (`bmw:1, byd:1, nissan_nv200:1`, plus K=2 edge cases): excluded from training (sampler filters len<K with log), kept in index/eval.
- Artifacts: `models/dino_triplet.pt` + `models/index.faiss` (or numpy fallback) + `models/index_meta.json` {backbone, method, dim, index_version, class list, threshold}.
- Inference: `CarEmbedder.embed_batch(pil_images: list) -> (B,D)` L2-normalized, cosine kNN k=5 majority vote → `{label, confidence, top5[{label, score}]}`. Single = batch of 1, identical path.
- Open-set add: two paths. (a) Index-add (daily): drop images in `car_brands/<new_class>/` → `build_index.py` light clean (corrupt/dedupe/resize) → seconds, no retrain. (b) Retrain (monthly/when ~10+ new shots per class or cross-view metric drops): full clean + resume triplet from last checkpoint + rebuild index.
- View handling (locked per discussion): class stays brand-type; view is auxiliary mining signal only, never splits classes. Query front matches front refs closest but votes same class; k=5 naturally contains both views.

## 4. Embedding-method abstraction (future-proofing, approved)

Embedding extraction must be swappable (CLS, patch-mean, CLS+concat, proj-256, future heads/ONNX) without API/frontend changes.

```python
class EmbedConfig: method: str  # "cls_proj256" (default, trainer output) | "cls" | "patch_mean" | "cls_patch_concat"
class CarEmbedder:
  def embed_batch(self, images: list[PIL]) -> Tensor  # (B,D), L2-normed
```

- `index_meta.json` records method+dim+backbone+version; loader refuses mismatched checkpoint/index with explicit error (fixes current 1536-d vs 256-d silent mismatch between `ViT/vision_transformers.py::DinoV2` and `training/modelloader.py::DinoV2Embedder`).
- Switching method = config change + rebuild + `eval.py` compare. API shape unchanged.

## 5. Dataset normalization

- Keep top-folder = class (user layout correct). Normalizer walks recursively, writes `metadata.csv` {filepath, class, view} where view inferred from `front/back/side/` subfolder name or `unknown`.
- Enforce class names lowercase `a-z0-9_`, dedupe by hash, drop corrupt/unreadable, log dropped 1-shot from train (kept in index).
- Eval split: view-disjoint where possible (hold out one view per class with >=2 views) to prove angle robustness; otherwise random holdout. Report same-view top-k and cross-view top-k separately.
- Fix required in existing `training/dataloader.py::VehicleDataset`: current `classes` derivation uses file parent (breaks with view subfolders); must use top-level folder relative to root.

## 6. Backend API (FastAPI, v1)

- `POST /v1/predict` — multipart single image → `{label, confidence, top5[], latency_ms, model_version}`.
- `POST /v1/predict/batch` — multipart up to 10 images, 8MB each → `{results: [...per image...]}`. Same handler internally: one `embed_batch` call (efficient batching, not a loop).
- `GET /v1/classes` — list of indexed classes + counts.
- `GET /health` — `{status, model_version, index_size, dim}`.
- Auth: `X-API-Key` required for API callers (manual issuance via `POST /admin/keys`, SQLite table). Demo web origin: no key but IP rate-limited (e.g. 20/day) + in-memory sliding window. All errors JSON `{detail}`: 400 bad image, 401 bad key, 413 too many/large, 429 rate-limited.
- Logging: request id, label, confidence, latency, key prefix. No image persistence by default (flag to enable for dataset collection later).
- Validation: Pillow verify, max dims, timeout, torch.no_grad, CPU default with CUDA if available.

## 7. Frontend demo (Next.js, v1)

- Single page: explicit **Upload button** (multi-select up to 10) + drag-drop zone + paste support + client downscale to 1024px before upload. Preview grid, per-image card: top-1 label + confidence bar + expandable top-5 + latency. `Try sample cars` button with 3 bundled examples. `Get API key — contact us` placeholder linking to backend admin (no self-signup in v1).
- Tailwind, `NEXT_PUBLIC_API_URL` env, fetch with progress + per-image error states (400/413/429 shown inline). No login. Mobile-friendly. Playwright smoke test: upload → results render.

## 8. ViT/ cleanup (approved, step 0 of implementation)

Keep: `training/{dataloader,modelloader,training}.py` (fix 3 bugs above + `__main__` syntax error + hardcoded paths).
Move to `tools/labeling/`: `naming_pipeline.py`, `prompt_mapping.py`, `brand_classifier.py` (fix self-import `from script.ViT...`, hardcoded paths).
Delete: `onnx_pipeline.py` (empty), `__pycache__/`, broken `requirements.txt` (replace), `CarBrandONNX` (20 US brands, hardcoded pakuwon path), duplicate `RealtimePipeline`/`InferencePipeline` CCTV loops (both import missing `car_embedding_db.chroma_db`, old `BASE_DIR.parents[2]`; merge YOLO detector later if needed), `add_image.py` (replaced by `build_index.py`).

## 9. Error handling

- Bad/corrupt image → 400 with filename. Oversize/count → 413. Bad key → 401. Rate-limited → 429 + `Retry-After`. Model/index version mismatch → 500 at startup with explicit message (fail fast, never silent wrong-dim search). Empty batch → 400. Per-image failures in batch return per-item `{error}` without failing whole batch.

## 10. Testing

- Backend pytest (CPU, mocked embedder + tiny random index): `embed_batch` shapes (1 vs 8 identical to singles loop), kNN voting ties, API 400/401/413/429 paths, method/dim mismatch refusal, `clean_dataset` view parsing.
- `eval.py`: held-out top-1/top-5 same-view and cross-view; triplet `active_triplets` diagnostic (collapse→0 = converged/slack, stuck ~100% + flat loss = not learning).
- Frontend: Playwright smoke (upload 1 + 3 batch → cards render, sample button works).
- No GPU in CI.

## 11. Non-goals (v1)

Stripe/payments, user accounts, billing portal, YOLO detection-before-classify in API (assume cropped car; detection later), ONNX export (method slot reserved), Postgres, multi-region deploy.

## 12. Open follow-ups (post-v1)

Collect demo uploads (opt-in) to fix imbalance; twin-group hard negatives (avanza/xenia/veloz) via `twin_groups`; ArcFace/SupCon comparison if triplet plateaus; view-conditioned threshold tuning.
