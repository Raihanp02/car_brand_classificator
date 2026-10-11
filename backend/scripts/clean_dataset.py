"""Dataset normalizer for the car classifier.

Walks ``car_brands/<class>[/<view>]/*.jpg`` and produces a ``metadata.csv``
with header::

    filepath,class,view,split,sha1,width,height,corrupt

Stdlib only (no torch / Pillow needed) so Task 1 tests run anywhere.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import struct
import sys
from pathlib import Path

VALID_VIEWS = {"front", "back", "side"}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def infer_view_and_class(relpath: str) -> tuple[str, str]:
    parts = Path(relpath).parts  # ('car_brands','<class>',...)
    cls = parts[1].lower()
    view = parts[2].lower() if len(parts) > 3 and parts[2].lower() in VALID_VIEWS else "unknown"
    return cls, view


def assign_split(sha1: str) -> str:
    """Deterministic 80/10/10 train/val/index split from the file hash."""
    bucket = int(sha1[:8], 16) % 10
    if bucket == 8:
        return "val"
    if bucket == 9:
        return "index"
    return "train"


def image_dimensions(path: Path) -> tuple[int | None, int | None]:
    """Read (width, height) from image headers without Pillow.

    Supports PNG, JPEG, GIF, BMP and WebP (VP8/VP8L/VP8X).
    Returns (None, None) when the dimensions cannot be determined.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(64)
            if len(head) < 12:
                return None, None
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                if len(head) >= 24 and head[12:16] == b"IHDR":
                    w, h = struct.unpack(">II", head[16:24])
                    return w, h
                return None, None
            if head[:6] in (b"GIF87a", b"GIF89a"):
                w, h = struct.unpack("<HH", head[6:10])
                return w, h
            if head[:2] == b"BM":
                w, h = struct.unpack("<ii", head[18:26])
                return w, abs(h)
            if head[:2] == b"\xff\xd8":
                return _jpeg_dimensions(f)
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                return _webp_dimensions(f)
    except OSError:
        return None, None
    return None, None


def _jpeg_dimensions(f) -> tuple[int | None, int | None]:
    f.seek(2)
    while True:
        marker_head = f.read(4)
        if len(marker_head) < 4 or marker_head[0] != 0xFF:
            return None, None
        marker = marker_head[1]
        if marker in (0xD8, 0xD9):  # SOI / EOI
            continue
        if 0xD0 <= marker <= 0xD7:  # RSTn: no length field
            continue
        seg_len = struct.unpack(">H", marker_head[2:4])[0]
        if seg_len < 2:
            return None, None
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            sof = f.read(max(seg_len - 2, 5))
            if len(sof) < 5:
                return None, None
            h, w = struct.unpack(">HH", sof[1:5])
            return w, h
        f.seek(seg_len - 2, 1)


def _webp_dimensions(f) -> tuple[int | None, int | None]:
    f.seek(12)
    chunk = f.read(16)
    if len(chunk) < 10:
        return None, None
    fourcc = chunk[:4]
    if fourcc == b"VP8 ":
        data = chunk[10:20] if len(chunk) >= 20 else chunk[10:] + f.read(20 - len(chunk))
        if len(data) < 10:
            return None, None
        w = struct.unpack("<H", data[6:8])[0] & 0x3FFF
        h = struct.unpack("<H", data[8:10])[0] & 0x3FFF
        return w, h
    if fourcc == b"VP8L":
        b = chunk[10:14] if len(chunk) >= 14 else chunk[10:] + f.read(14 - len(chunk))
        if len(b) < 4 or b[0] != 0x2F:
            return None, None
        bits = struct.unpack("<I", b)[0] >> 8
        w = (bits & 0x3FFF) + 1
        h = ((bits >> 14) & 0x3FFF) + 1
        return w, h
    if fourcc == b"VP8X":
        data = chunk[10:20] if len(chunk) >= 20 else chunk[10:] + f.read(20 - len(chunk))
        if len(data) < 10:
            return None, None
        w = int.from_bytes(data[4:7], "little") + 1
        h = int.from_bytes(data[7:10], "little") + 1
        return w, h
    return None, None


def sha1_of(path: Path) -> str | None:
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def collect_rows(root: Path) -> list[dict]:
    """Scan *root* and return one metadata row per image file.

    Rows with a duplicate sha1 (CCTV copies across views/classes) are dropped,
    keeping the first occurrence; the dropped count is logged. ``sha1`` is
    still recorded on every kept row.
    """
    try:
        anchor = root.parent
        prefix = root.name
    except Exception:  # pragma: no cover - defensive
        anchor = Path(".")
        prefix = root.name
    rows: list[dict] = []
    seen: set[str] = set()
    dropped = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        try:
            rel = path.relative_to(anchor)
        except ValueError:
            rel = Path(prefix) / path.relative_to(root)
        relpath = rel.as_posix()
        cls, view = infer_view_and_class(relpath)
        digest = sha1_of(path)
        if digest is not None:
            if digest in seen:
                dropped += 1
                continue
            seen.add(digest)
        width, height = image_dimensions(path)
        corrupt = 0 if digest is not None and width is not None else 1
        rows.append(
            {
                "filepath": relpath,
                "class": cls,
                "view": view,
                "split": assign_split(digest) if digest else "train",
                "sha1": digest or "",
                "width": width if width is not None else "",
                "height": height if height is not None else "",
                "corrupt": corrupt,
            }
        )
    if dropped:
        print(f"dedupe: dropped {dropped} duplicate-sha1 rows (kept first)")
    return rows


HEADER = ["filepath", "class", "view", "split", "sha1", "width", "height", "corrupt"]


def print_summary(rows: list[dict]) -> None:
    classes = sorted({r["class"] for r in rows})
    print(f"classes={len(classes)} images={len(rows)}")
    per_class: dict[str, int] = {}
    for r in rows:
        per_class[r["class"]] = per_class.get(r["class"], 0) + 1
    for cls in classes:
        print(f"{cls}: {per_class[cls]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize car_brands/ into metadata.csv")
    parser.add_argument("--root", default="car_brands")
    parser.add_argument("--out", default="metadata.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    rows = collect_rows(root)
    print_summary(rows)
    if args.dry_run:
        return 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
