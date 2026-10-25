"""ONNX Runtime embedder: preprocessing + session.run + L2 normalization.

The graph contract is fixed by the training repo:

    input:  pixel_values  float32 [N, 3, 224, 224]
    output: embeddings    float32 [N, D]  (L2-normalized)

Preprocessing parameters are read from ``model.meta.json`` rather than
hardcoded, so a contract change cannot silently drift from the exporter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.ml.contract import ModelContract


def preprocess_image(image: Image.Image, preprocessing: dict[str, Any]) -> np.ndarray:
    """Contract-defined preprocessing: RGB -> resize -> scale -> normalize -> CHW."""
    resize = tuple(int(v) for v in preprocessing.get("resize", (224, 224)))
    mean = np.asarray(preprocessing.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(preprocessing.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    scale = float(preprocessing.get("scale", 1.0 / 255.0))
    img = image.convert("RGB").resize(resize, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) * scale
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))


class OnnxEmbedder:
    def __init__(
        self,
        model_path: str | Path,
        contract: ModelContract,
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort

        self.contract = contract
        self.dim = contract.dim
        self.session = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError(f"ONNX model has no inputs: {model_path}")
        self.input_name = inputs[0].name

        self._preprocessing: dict[str, Any] = contract.preprocessing

    def preprocess(self, image: Image.Image) -> np.ndarray:
        return preprocess_image(image, self._preprocessing)

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        batch = np.stack([self.preprocess(im) for im in images]).astype(np.float32)
        out = np.asarray(
            self.session.run(None, {self.input_name: batch})[0], dtype=np.float32
        )
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        return out / np.maximum(norm, 1e-12)
