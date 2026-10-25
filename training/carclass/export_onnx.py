"""Export a trained CarEmbedder checkpoint to ONNX for the backend.

The exported graph contains backbone + embedding head and returns
L2-normalized embeddings. Preprocessing stays outside the graph; its exact
parameters are written to ``model.meta.json`` so the backend can reproduce it.

Usage::

    uv run python -m carclass.export_onnx \
        --checkpoint dinov2_triplet.pt --method cls_proj256 \
        --out-dir artifacts --model-version registers-base_triplet_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from carclass.embedder import _IMAGENET_MEAN, _IMAGENET_STD, BACKBONE, CarEmbedder
from carclass.heads import DEFAULT_METHOD, METHODS

CONTRACT_VERSION = 1
INPUT_SIZE = 224


class _ExportWrapper(torch.nn.Module):
    """Backbone + head, matching CarEmbedder.embed_batch exactly."""

    def __init__(self, embedder: CarEmbedder):
        super().__init__()
        self.backbone = embedder.backbone
        self.proj = embedder.proj

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
        emb = self.proj(hidden)
        return emb / emb.norm(dim=-1, keepdim=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _verify(model_path: Path, embedder: CarEmbedder) -> float:
    import numpy as np
    import onnxruntime as ort

    dummy = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        expected = _ExportWrapper(embedder)(dummy).numpy()
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    actual = sess.run(None, {input_name: dummy.numpy()})[0]
    cos = np.sum(expected * actual, axis=1) / (
        np.linalg.norm(expected, axis=1) * np.linalg.norm(actual, axis=1)
    )
    return float(cos.min())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export CarEmbedder to ONNX")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--method", default=DEFAULT_METHOD, choices=sorted(METHODS))
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--model-name", default="model.onnx")
    ap.add_argument("--model-version", default="registers-base_triplet_v1")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / args.model_name
    meta_path = out_dir / "model.meta.json"

    embedder = CarEmbedder(
        backbone=args.backbone, method=args.method, checkpoint=args.checkpoint
    )
    wrapper = _ExportWrapper(embedder).eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(model_path),
            input_names=["pixel_values"],
            output_names=["embeddings"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "embeddings": {0: "batch"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )

    meta = {
        "contract_version": CONTRACT_VERSION,
        "model_version": args.model_version,
        "backbone": args.backbone,
        "method": args.method,
        "dim": embedder.dim,
        "opset": args.opset,
        "preprocessing": {
            "resize": [INPUT_SIZE, INPUT_SIZE],
            "interpolation": "bilinear",
            "color": "RGB",
            "mean": _IMAGENET_MEAN,
            "std": _IMAGENET_STD,
            "layout": "NCHW",
            "scale": 1.0 / 255.0,
        },
        "sha256": _sha256(model_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {model_path} ({embedder.dim}-d, method={args.method})")
    print(f"wrote {meta_path}")

    if not args.skip_verify:
        try:
            cosine = _verify(model_path, embedder)
        except ImportError as exc:
            print(f"skipped verification (onnxruntime missing: {exc})", file=sys.stderr)
        else:
            print(f"torch-vs-onnx min cosine: {cosine:.6f}")
            if cosine < 0.999:
                print("PARITY TOO LOW: check preprocessing/head", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
