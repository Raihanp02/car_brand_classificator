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

from app.ml.contract import ModelContract, verify_model_file
from config import settings

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
        batch_size: int = 16,
    ):
        import onnxruntime as ort

        self.batch_size = batch_size if batch_size else settings.BATCH_SIZE
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

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        contract: ModelContract,
        providers: list[str] | None = None,
    ) -> "OnnxEmbedder":
        """Verify the file against the contract, load it, and sanity-check output."""
        verify_model_file(model_path, contract)        # sha256 pin
        emb = cls(model_path, contract, providers)

        probe = Image.new("RGB", (64, 64))             # dummy image, also warms up the session
        out = emb.embed_batch([probe])
        if out.shape != (1, contract.dim):
            raise ValueError(
                f"embedder output {out.shape} disagrees with contract dim={contract.dim}"
            )
        return emb

    def preprocess(self, image: Image.Image) -> np.ndarray:
        return preprocess_image(image, self._preprocessing)

    def embed_batch(
        self,
        images: list[Image.Image],
        batch_size: int | None = None,
    ) -> np.ndarray:
        batch_size = batch_size or self.batch_size  # e.g. set in __init__, default 32
        if not images:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        outputs = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            batch = np.stack([self.preprocess(im) for im in chunk]).astype(np.float32)
            out = np.asarray(
                self.session.run(None, {self.input_name: batch})[0], dtype=np.float32
            )
            outputs.append(out)

        out = np.concatenate(outputs, axis=0)
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        return out / np.maximum(norm, 1e-12)
