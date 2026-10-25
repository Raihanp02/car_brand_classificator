"""Repo-relative path resolution shared by training entrypoints."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def dataset_path(rel: str | Path) -> Path:
    """Resolve a metadata filepath, allowing both repo-root and data/ layouts."""
    p = Path(rel)
    if p.is_absolute():
        return p
    for base in (REPO_ROOT, REPO_ROOT / "data"):
        candidate = base / p
        if candidate.exists():
            return candidate
    return REPO_ROOT / p
