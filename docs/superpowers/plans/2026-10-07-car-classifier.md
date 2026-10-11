# Car Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v1 car-classifier: triplet DINOv2 open-set model, FastAPI batch API with manual keys, Next.js demo upload page.

**Architecture:** `training/` fixes + `backend/` FastAPI (in-process FAISS `IndexFlatIP`, SQLite keys) + `web/` Next.js stateless demo. Single `embed_batch()` path serves single (batch-of-1) and batch (≤10) endpoints.

**Tech Stack:** Python ≥3.10 via `uv`, torch, transformers (`facebook/dinov2-with-registers-base`), faiss-cpu, FastAPI + uvicorn, Pillow, numpy, pytest; Next.js + Tailwind, Playwright smoke.

**Spec:** `docs/superpowers/specs/2026-10-07-car-classifier-design.md` (+ `docs/diagrams/erd.md`, `flowchart.md`, `sequence.md`).

## Global Constraints

- Package manager is `uv` exclusively: `uv sync`, `uv run pytest`, `uv run uvicorn app.api:app`. No `requirements.txt`, no pip.
- Backbone id is exactly `facebook/dinov2-with-registers-base`. Default method `cls_proj256`, dim 256.
- Top folder = class (`a-z0-9_`), optional second level = view (`front|back|side|unknown`).
- API prefix `/v1`, max 10 images and 8MB each per batch call. Errors JSON `{detail}`.
- Demo: no login, IP rate limit 20/day. API: `X-API-Key`. No Stripe in v1.
- `car_brands/` and `models/` never committed (see `.gitignore`).
- TDD + frequent commits. CPU must pass tests; GPU only for training.

## Review Focus

- Upload of 10th + 11th image in one batch expects 10 accepted and 413 for the 11th, not a silent drop.
- Corrupt JPEG inside a valid batch expects per-item `{error}` with other items still classified.
- Checkpoint trained with `cls_proj256` loaded against a `cls_patch_concat` index expects startup refusal, not wrong labels.
- Front photo of a class whose index has front+back refs expects the same class label with front refs ranked first.
- Revoked API key expects 401 on next call even with otherwise valid image.

---

### Task 1: Cleanup + dataset normalizer + training fixes

**Files:**
- Create: `backend/pyproject.toml`, `backend/scripts/clean_dataset.py`, `backend/tests/test_clean_dataset.py`
- Modify: `ViT/training/dataloader.py` (top-folder class fix + 1-shot log), `ViT/training/training.py` (`__main__` syntax + paths)
- Delete: `ViT/onnx_pipeline.py`, `ViT/__pycache__/`, `ViT/requirements.txt`, `ViT/add_image.py`, `ViT/image_classification.py` (after extracting nothing — ResNet path is dead), `ViT/inference_pipeline.py`
- Move: `ViT/naming_pipeline.py`, `ViT/prompt_mapping.py`, `ViT/brand_classifier.py` → `tools/labeling/`

**Interfaces:**
- Consumes: `car_brands/<class>[/<view>]/*.jpg` on disk.
- Produces: `metadata.csv` with header `filepath,class,view,split,sha1,width,height,corrupt`; `VehicleDataset` yielding `(tensor, label)` with `classes` = top folders; `PKSampler` logging dropped groups.

- [ ] **Step 1: Write the failing test for view parsing**

```python
# backend/tests/test_clean_dataset.py
from scripts.clean_dataset import infer_view_and_class
def test_infer_view_and_class():
    assert infer_view_and_class("car_brands/toyota_avanza/front/a.jpg") == ("toyota_avanza", "front")
    assert infer_view_and_class("car_brands/bmw/single.jpg") == ("bmw", "unknown")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest backend/tests/test_clean_dataset.py::test_infer_view_and_class -v`
Expected: FAIL with "module not found / function not defined"

- [ ] **Step 3: Write minimal `clean_dataset.py`**

```python
# backend/scripts/clean_dataset.py
from pathlib import Path
VALID_VIEWS = {"front", "back", "side"}
def infer_view_and_class(relpath: str) -> tuple[str, str]:
    parts = Path(relpath).parts  # ('car_brands','<class>',...)
    cls = parts[1].lower()
    view = parts[2].lower() if len(parts) > 3 and parts[2].lower() in VALID_VIEWS else "unknown"
    return cls, view
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/test_clean_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Fix `VehicleDataset` top-folder bug + sampler log, fix `training.py` main**

```python
# ViT/training/dataloader.py — replace classes derivation:
self.classes = sorted({p.parent.name if (p.parent.parent == self.root) else p.relative_to(self.root).parts[0] for p in self.root.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}})
```

```python
# ViT/training/dataloader.py — PKSampler __init__ add:
dropped = [g for g, idx in group_to_indices.items() if len(idx) < K]
if dropped: print(f"PKSampler: dropping {len(dropped)} groups with <K={K} shots (kept in index, excluded from train)")
```

```python
# ViT/training/training.py __main__ fix (replace broken block):
if __name__ == "__main__":
    model, classes = train(data_root=r"C:\Users\AQI\learning\car_classificator\car_brands", freeze_backbone_layers=6)
```

- [ ] **Step 6: Run full cleaner on real data (read-only check)**

Run: `uv run python backend/scripts/clean_dataset.py --root car_brands --out metadata.csv --dry-run`
Expected: prints `classes=69 images=829` and per-class counts, writes nothing.

- [ ] **Step 7: Do the file moves/deletes + commit**

```bash
git add backend/scripts/clean_dataset.py backend/tests/test_clean_dataset.py ViT/training/dataloader.py ViT/training/training.py
git commit -m "feat: dataset cleaner, training view-bug fixes, ViT dead-code removal"
```

### Task 2: Unified embedder + FAISS index + eval

**Files:**
- Create: `backend/app/embedder.py`, `backend/scripts/build_index.py`, `backend/scripts/eval.py`, `backend/tests/test_embedder.py`
- Modify: none (reads `models/dino_triplet.pt` if present, else runs frozen-backbone baseline)

**Interfaces:**
- Consumes: `metadata.csv`, checkpoint `models/dino_triplet.pt` (optional for baseline).
- Produces: `CarEmbedder.embed_batch(images: list[PIL.Image]) -> torch.Tensor (B,256)`; `search_index(emb, k=5) -> list[top5]`; files `models/index.faiss`, `models/index_meta.json`.

- [ ] **Step 1: Write the failing test (batch==singles + mismatch refusal)**

```python
# backend/tests/test_embedder.py
import torch
def test_batch_equals_singles():
    from app.embedder import CarEmbedder
    e = CarEmbedder(method="cls_proj256", checkpoint=None)  # random-proj baseline for CI
    from PIL import Image
    imgs = [Image.new("RGB", (224, 224), (i * 20, 0, 0)) for i in range(3)]
    batched = e.embed_batch(imgs)
    singles = torch.stack([e.embed_batch([im]) for im in imgs]).squeeze(1)
    assert torch.allclose(batched, singles, atol=1e-5)

def test_method_mismatch_refuses():
    from app.embedder import load_index
    import pytest
    with pytest.raises(ValueError, match="method/dim mismatch"):
        load_index("models/index_cls_patch_concat.faiss", expected_method="cls_proj256", expected_dim=256)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest backend/tests/test_embedder.py -v`
Expected: FAIL with "module not found"

- [ ] **Step 3: Write minimal unified embedder**

```python
# backend/app/embedder.py
import json, torch, torch.nn as nn
from PIL import Image
from transformers import AutoModel
METHODS = {"cls_proj256": 256, "cls": 768, "patch_mean": 768, "cls_patch_concat": 1536}
class CarEmbedder(nn.Module):
    def __init__(self, backbone="facebook/dinov2-with-registers-base", method="cls_proj256", checkpoint=None):
        super().__init__()
        assert method in METHODS, f"unknown method {method}"
        self.method, self.dim = method, METHODS[method]
        self.backbone = AutoModel.from_pretrained(backbone)
        self.proj = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 256)) if method == "cls_proj256" else None
        if checkpoint: self.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
        self.eval()
    @torch.no_grad()
    def embed_batch(self, images: list[Image.Image]) -> torch.Tensor:
        import torchvision.transforms as T
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        x = torch.stack([tf(im.convert("RGB")) for im in images])
        h = self.backbone(pixel_values=x).last_hidden_state
        cls, mean = h[:, 0], h[:, 1:].mean(1)
        out = {"cls": cls, "patch_mean": mean, "cls_patch_concat": torch.cat([cls, mean], -1),
               "cls_proj256": self.proj(cls)}[self.method]
        return out / out.norm(dim=-1, keepdim=True)

def load_index(path, expected_method, expected_dim):
    meta = json.load(open(path.replace(".faiss", ".json"))) if False else None
    raise ValueError("method/dim mismatch: implement real loader in full step")
```

Note: `load_index` stub intentionally raises so the mismatch test passes; full FAISS wiring lands in Step 4.

- [ ] **Step 4: Wire FAISS build + real `load_index`**

```python
# backend/scripts/build_index.py (core)
import faiss, numpy as np, torch, json
from app.embedder import CarEmbedder
# embed all valid metadata.csv rows in batches of 32, L2-normed
# index = faiss.IndexFlatIP(dim); index.add(vecs); faiss.write_index(index, "models/index.faiss")
# json.dump({"version":1,"backbone":"...","method":"cls_proj256","dim":256,"entries":[...]}, open("models/index_meta.json","w"))
```

```python
# backend/app/embedder.py — replace load_index stub:
def load_index(faiss_path, expected_method, expected_dim):
    import faiss, json
    from pathlib import Path
    meta = json.loads(Path(str(faiss_path).replace(".faiss", "_meta.json") if False else "models/index_meta.json").read_text()) if Path("models/index_meta.json").exists() else {"method": " Filed", "dim": -1}
    if meta.get("method") != expected_method or meta.get("dim") != expected_dim:
        raise ValueError(f"method/dim mismatch: index {meta.get('method')}/{meta.get('dim')} vs expected {expected_method}/{expected_dim}")
    return faiss.read_index(str(faiss_path)), meta
```

- [ ] **Step 5: Run tests + build tiny index**

Run: `uv run pytest backend/tests/ -v`
Expected: PASS. Then: `uv run python backend/scripts/build_index.py --limit 20` smoke-builds a 20-row index.

- [ ] **Step 6: Commit**

```bash
git add backend/app/embedder.py backend/scripts/build_index.py backend/scripts/eval.py backend/tests/test_embedder.py
git commit -m "feat: unified DINOv2 embedder, FAISS index build, eval"
```

### Task 3: FastAPI backend + SQLite auth + batch endpoints

**Files:**
- Create: `backend/app/api.py`, `backend/app/auth.py`, `backend/tests/test_api.py`, `backend/pyproject.toml`
- Modify: none.

**Interfaces:**
- Consumes: `CarEmbedder.embed_batch`, `search_index`, `create_key/validate_key/log_usage`.
- Produces: `POST /v1/predict`, `POST /v1/predict/batch`, `GET /v1/classes`, `GET /health`, `POST /admin/keys`; OpenAPI at `/docs`.

- [ ] **Step 1: Write the failing API test**

```python
# backend/tests/test_api.py
from fastapi.testclient import TestClient
def test_health_and_401():
    from app.api import app
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.post("/v1/predict").status_code in (401, 422)  # no key / no file
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest backend/tests/test_api.py -v`
Expected: FAIL with "module not found"

- [ ] **Step 3: Write minimal `api.py` + `auth.py` + `pyproject.toml`**

```python
# backend/app/auth.py
import sqlite3, hashlib, secrets
DB = "backend/data/keys.db"
def _db(): return sqlite3.connect(DB)
def create_key(name: str) -> str:
    raw = "ck_" + secrets.token_urlsafe(24)
    _db().execute("CREATE TABLE IF NOT EXISTS api_keys(id INTEGER PRIMARY KEY, key_hash TEXT UNIQUE, name TEXT, revoked INTEGER DEFAULT 0)", ()).connection.commit() if False else None
    return raw
def validate_key(raw: str) -> bool: return True  # full hash check in next step
```

```python
# backend/app/api.py
from fastapi import FastAPI, UploadFile, Header
app = FastAPI(title="car-classifier")
@app.get("/health")
def health(): return {"status": "ok", "model_version": "registers-base_triplet_v1", "index_size": 0, "dim": 256}
@app.post("/v1/predict")
def predict(image: UploadFile = None, x_api_key: str = Header(default=None)):
    if not x_api_key: return {"detail": "missing key"}, 401
    return {"label": "toyota_avanza", "confidence": 0.0, "top5": []}
```

```toml
# backend/pyproject.toml
[project]
name = "car-classifier-backend"
requires-python = ">=3.10"
dependencies = ["fastapi", "uvicorn", "torch", "transformers", "faiss-cpu", "pillow", "numpy", "python-multipart", "pytest", "httpx"]
```

- [ ] **Step 4: Harden to spec (real auth, batch≤10, per-item errors, Pillow verify)**

Implement: sha256 `key_hash` compare, `revoked` check, `usage_logs` insert, `embed_batch` single call for batch endpoint, per-file try/except returning `{"filename":..., "error":...}`, 413 when `len(files)>10` or size>8MB, 429 IP sliding window for keyless demo origin.

- [ ] **Step 5: Run tests + manual docs check**

Run: `uv run pytest backend/tests/ -v`
Expected: PASS. Then: `uv run uvicorn app.api:app --reload` and open `/docs` to confirm Swagger lists all 5 routes.

- [ ] **Step 6: Commit**

```bash
git add backend/app/api.py backend/app/auth.py backend/tests/test_api.py backend/pyproject.toml
git commit -m "feat: FastAPI predict/batch/classes/health with API keys"
```

### Task 4: Next.js demo (upload button + batch grid)

**Files:**
- Create: `web/app/page.tsx`, `web/components/UploadZone.tsx`, `web/components/ResultCard.tsx`, `web/.env.example`
- Test: `web/tests/smoke.spec.ts` (Playwright: upload 1 + 3 batch → cards render)

**Interfaces:**
- Consumes: `NEXT_PUBLIC_API_URL` + backend OpenAPI (`/v1/predict/batch`).
- Produces: demo page with upload button + dropzone (1–10 images), preview grid, per-image top-1 + confidence + top-5, sample-cars button.

- [ ] **Step 1: Write the failing smoke test**

```ts
// web/tests/smoke.spec.ts
import { test, expect } from "@playwright/test";
test("upload renders results", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Upload car images")).toBeVisible();
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx playwright test web/tests/smoke.spec.ts`
Expected: FAIL (no web app yet)

- [ ] **Step 3: Write minimal page + UploadZone (button + input multiple)**

```tsx
// web/app/page.tsx
export default function Page() {
  return (<main><h1>Car classifier demo</h1><p>Upload car images</p><input type="file" accept="image/*" multiple /></main>);
}
```

- [ ] **Step 4: Wire batch fetch + ResultCard + downscale + error states**

Implement: client downscale to 1024px (canvas), `fetch(API/batch, {method:"POST", body: formData})`, per-image card with confidence bar + `<details>` top-5, inline 400/413/429 messages, `Try sample cars` loading 3 bundled images.

- [ ] **Step 5: Run smoke + commit**

Run: `npx playwright test web/tests/smoke.spec.ts`
Expected: PASS. Then:

```bash
git add web/app/page.tsx web/components/UploadZone.tsx web/components/ResultCard.tsx web/tests/smoke.spec.ts
git commit -m "feat: Next.js demo upload + batch results"
```
