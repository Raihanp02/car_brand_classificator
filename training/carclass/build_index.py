"""Build the FAISS search index from metadata.csv rows.

Embeds all valid (corrupt == 0) rows in batches of 32 with
:class:`carclass.embedder.CarEmbedder` (L2-normalized), stores them in a
``faiss.IndexFlatIP`` index, and writes ``artifacts/index.faiss`` plus the
``artifacts/index_meta.json`` sidecar::

    {"version": 1, "backbone": ..., "method": "cls_proj256", "dim": 256,
     "entries": [{"filepath": ..., "class": ..., "view": ..., "split": ...}]}

Usage (from repo root)::

    uv run python -m carclass.build_index --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from carclass.paths import REPO_ROOT, dataset_path


def resolve_metadata(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    alt = REPO_ROOT / path
    if alt.exists():
        return alt
    return p  # let the caller report the missing file


def main(argv: list[str] | None = None) -> int:
    from carclass.embedder import BACKBONE, DEFAULT_METHOD, METHODS, CarEmbedder

    ap = argparse.ArgumentParser(description="Build FAISS index from metadata.csv")
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--method", default=DEFAULT_METHOD, choices=sorted(METHODS))
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="index only the first N valid rows (smoke builds)")
    args = ap.parse_args(argv)

    import faiss
    import numpy as np
    import torch
    from PIL import Image

    meta_path = resolve_metadata(args.metadata)
    if not meta_path.exists():
        print(f"metadata not found: {args.metadata}", file=sys.stderr)
        return 2

    with open(meta_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    valid = [r for r in rows if str(r.get("corrupt", 0)) == "0"]
    if args.limit is not None:
        valid = valid[: args.limit]
    if not valid:
        print("no valid rows to index", file=sys.stderr)
        return 2

    embedder = CarEmbedder(
        backbone=args.backbone, method=args.method, checkpoint=args.checkpoint
    )
    dim = embedder.dim

    vecs: list[np.ndarray] = []
    entries: list[dict] = []
    missing = 0
    for i in range(0, len(valid), args.batch_size):
        batch = valid[i : i + args.batch_size]
        imgs, keep = [], []
        for r in batch:
            fp = dataset_path(r["filepath"])
            if not fp.exists():
                missing += 1
                continue
            try:
                imgs.append(Image.open(fp).convert("RGB"))
                keep.append(r)
            except Exception:
                missing += 1
        if not imgs:
            continue
        with torch.no_grad():
            e = embedder.embed_batch(imgs).cpu().numpy().astype(np.float32)
        vecs.append(e)
        entries.extend(
            {
                "filepath": r["filepath"],
                "class": r["class"],
                "view": r.get("view", "unknown"),
                "split": r.get("split", "train"),
            }
            for r in keep
        )

    if not vecs:
        print("no embeddable images found", file=sys.stderr)
        return 2
    mat = np.concatenate(vecs, axis=0)
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = out_dir / "index.faiss"
    meta_out = out_dir / "index_meta.json"
    faiss.write_index(index, str(faiss_path))
    meta_out.write_text(
        json.dumps(
            {
                "version": 1,
                "backbone": args.backbone,
                "method": args.method,
                "dim": dim,
                "entries": entries,
            },
            indent=2,
        )
    )
    print(
        f"indexed {len(entries)} rows (skipped {missing} missing/unreadable)"
        f" -> {faiss_path}, {meta_out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
