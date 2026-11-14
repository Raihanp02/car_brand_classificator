"""Static configuration and artifact paths for the serving API.

The backend loads only ONNX + FAISS artifacts produced by the training repo;
it never imports torch/transformers. Paths resolve from the backend directory,
never cwd-relative, and can be overridden for deployments.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    ARTIFACTS_DIR: Path = BACKEND_DIR / "artifacts"
    MODEL_PATH: Path | None = None
    MODEL_META_PATH: Path | None = None
    INDEX_PATH: Path | None = None
    INDEX_META_PATH: Path | None = None

    # Upload limits
    MAX_FILES: int = Field(10, gt=0)
    MAX_BYTES: int = Field(8 * 1024 * 1024, gt=0)  # 8MB per image
    MAX_DIM: int = Field(8000, gt=0)  # max width/height in px

    # Search
    TOP_K: int = Field(5, gt=0)

    # Demo
    DEMO_WINDOW_S: int = Field(3600, gt=0)  # keyless demo sliding window
    DEMO_LIMIT_PER_TIME: int = Field(100, gt=0)

    # Image
    BATCH_SIZE: int = Field(16, gt=0)  # batch size for embedding model

    @model_validator(mode="after")
    def _derive_artifact_paths(self) -> "Settings":
        # Unset paths default to files inside ARTIFACTS_DIR,
        # so overriding ARTIFACTS_DIR moves everything with it.
        self.MODEL_PATH = self.MODEL_PATH or self.ARTIFACTS_DIR / "model.onnx"
        self.MODEL_META_PATH = self.MODEL_META_PATH or self.ARTIFACTS_DIR / "model.meta.json"
        self.INDEX_PATH = self.INDEX_PATH or self.ARTIFACTS_DIR / "index.faiss"
        self.INDEX_META_PATH = self.INDEX_META_PATH or self.ARTIFACTS_DIR / "index_meta.json"
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
