"""Train/serve artifact contract tests.

Artifact-dependent tests skip on a clean checkout (the ONNX file is huge and
gitignored); they run in CI/deploy where artifacts are published.
"""

import json

import numpy as np
import pytest
from PIL import Image

from app import config
from app.ml.contract import load_contract
from app.ml.onnx_embedder import preprocess_image

PREPROCESSING = {
    "resize": [224, 224],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "scale": 1 / 255,
    "layout": "NCHW",
}


def _write_meta(tmp_path, **overrides):
    meta = {
        "contract_version": 1,
        "model_version": "v1",
        "method": "cls_proj256",
        "dim": 256,
        "preprocessing": PREPROCESSING,
    }
    meta.update(overrides)
    path = tmp_path / "model.meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_load_contract_roundtrip(tmp_path):
    contract = load_contract(_write_meta(tmp_path))
    assert contract.method == "cls_proj256"
    assert contract.dim == 256
    assert contract.preprocessing["resize"] == [224, 224]


def test_load_contract_missing_keys_refuses(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"method": "cls"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        load_contract(path)


def test_preprocess_is_nchw_float32():
    arr = preprocess_image(Image.new("RGB", (640, 480), (255, 0, 0)), PREPROCESSING)
    assert arr.shape == (3, 224, 224)
    assert arr.dtype == np.float32


def test_published_contract_matches_index():
    if not config.MODEL_META_PATH.exists() or not config.INDEX_META_PATH.exists():
        pytest.skip("artifacts not present")
    contract = load_contract(config.MODEL_META_PATH)
    from app.ml.index import load_index

    index, meta = load_index(config.INDEX_PATH, contract.method, contract.dim)
    assert index.ntotal > 0
    assert meta["dim"] == contract.dim
