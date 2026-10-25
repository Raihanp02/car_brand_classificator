"""Static configuration and artifact paths for the serving API.

The backend loads only ONNX + FAISS artifacts produced by the training repo;
it never imports torch/transformers. Paths resolve from the backend directory,
never cwd-relative, and can be overridden for deployments.
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", BACKEND_DIR / "artifacts"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ARTIFACTS_DIR / "model.onnx"))
MODEL_META_PATH = Path(os.environ.get("MODEL_META_PATH", ARTIFACTS_DIR / "model.meta.json"))
INDEX_PATH = Path(os.environ.get("INDEX_PATH", ARTIFACTS_DIR / "index.faiss"))
INDEX_META_PATH = Path(os.environ.get("INDEX_META_PATH", ARTIFACTS_DIR / "index_meta.json"))

MAX_FILES = 10
MAX_BYTES = 8 * 1024 * 1024  # 8MB per image
MAX_DIM = 8000  # max width/height in px
TOP_K = 5
DEMO_WINDOW_S = 24 * 3600  # keyless demo sliding window
