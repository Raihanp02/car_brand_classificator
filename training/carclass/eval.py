"""Retrieval eval for the FAISS index (car-classifier v1).

Embeds query rows from one metadata split, searches the on-disk index with
:meth:`carclass.embedder.search_index`, and reports top-1 / top-5 same-class
accuracy. Hits on the query's own filepath are skipped (leave-one-out), so
evaluating a split that is also indexed does not inflate scores.

Usage (from repo root)::

    uv run python -m carclass.eval --split val --limit 200
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from carclass.paths import REPO_ROOT, dataset_path


def main(argv: list[str] | None = None) -> int:
    from carclass.embedder import (
        BACKBONE,
        DEFAULT_METHOD,
        METHODS,
        CarEmbedder,
        load_index,
    )

    ap = argparse.ArgumentParser(description="Eval FAISS retrieval accuracy")
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--index", default="artifacts/index.faiss")
    ap.add_argument("--split", default="val", choices=["train", "val", "index"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--method", default=DEFAULT_METHOD, choices=sorted(METHODS))
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    from PIL import Image

    meta_path = Path(args.metadata)
    if not meta_path.exists() and (REPO_ROOT / args.metadata).exists():
        meta_path = REPO_ROOT / args.metadata
    if not meta_path.exists():
        print(f"metadata not found: {args.metadata}", file=sys.stderr)
        return 2
    with open(meta_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    queries = [r for r in rows if r.get("split") == args.split and str(r.get("corrupt", 0)) == "0"]
    if args.limit is not None:
        queries = queries[: args.limit]
    if not queries:
        print(f"no query rows for split={args.split}", file=sys.stderr)
        return 2

    index_path = Path(args.index)
    if not index_path.exists() and (REPO_ROOT / args.index).exists():
        index_path = REPO_ROOT / args.index
    index, meta = load_index(
        index_path, expected_method=args.method, expected_dim=METHODS[args.method]
    )
    entries = meta.get("entries", [])
    by_path = {e["filepath"]: e for e in entries}

    embedder = CarEmbedder(
        backbone=args.backbone, method=args.method, checkpoint=args.checkpoint
    )

    top1 = topk = n = 0
    sv_top1 = sv_topk = sv_n = 0
    xv_top1 = xv_topk = xv_n = 0
    for i in range(0, len(queries), args.batch_size):
        batch = queries[i : i + args.batch_size]
        imgs, keep = [], []
        for r in batch:
            fp = dataset_path(r["filepath"])
            if not fp.exists():
                continue
            try:
                imgs.append(Image.open(fp).convert("RGB"))
                keep.append(r)
            except Exception:
                continue
        if not imgs:
            continue
        with torch.no_grad():
            q = embedder.embed_batch(imgs).cpu().numpy().astype(np.float32)
        # +1 neighbour so self-match exclusion still leaves k candidates.
        scores, ids = index.search(q, args.k + 1)
        for qi, r in enumerate(keep):
            hits = [
                entries[j]
                for j in ids[qi]
                if 0 <= j < len(entries) and entries[j].get("filepath") != r["filepath"]
            ][: args.k]
            if not hits:
                continue
            # Same-view query: another ref of the same class+view exists in
            # the index; otherwise the query must match across views.
            same_view = any(
                e.get("filepath") != r["filepath"]
                and e.get("class") == r["class"]
                and e.get("view", "unknown") == r.get("view", "unknown")
                for e in entries
            )
            n += 1
            hit1 = hits[0].get("class") == r["class"]
            hitk = any(h.get("class") == r["class"] for h in hits)
            if hit1:
                top1 += 1
            if hitk:
                topk += 1
            if same_view:
                sv_n += 1
                sv_top1 += hit1
                sv_topk += hitk
            else:
                xv_n += 1
                xv_top1 += hit1
                xv_topk += hitk

    if not n:
        print("no evaluable queries", file=sys.stderr)
        return 2
    print(
        f"split={args.split} n={n} top1={top1 / n:.3f} top{args.k}={topk / n:.3f}"
        f" ({top1}/{n}, {topk}/{n})"
    )
    if sv_n:
        print(
            f"same_view n={sv_n} top1={sv_top1 / sv_n:.3f} top{args.k}={sv_topk / sv_n:.3f}"
            f" ({sv_top1}/{sv_n}, {sv_topk}/{sv_n})"
        )
    else:
        print("same_view n=0 (no queries with a same-view ref)")
    if xv_n:
        print(
            f"cross_view n={xv_n} top1={xv_top1 / xv_n:.3f} top{args.k}={xv_topk / xv_n:.3f}"
            f" ({xv_top1}/{xv_n}, {xv_topk}/{xv_n})"
        )
    else:
        print("cross_view n=0 (every query has a same-view ref)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
