"""Request/response schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

class ClassScore(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)

class ImageResult(BaseModel):
    filename: str
    label: str | None = None                     # top-1 class
    confidence: float | None = Field(default=None, ge=0, le=1)
    top_k: list[ClassScore] | None = None        # optional, top-k alternatives
    error: str | None = None                     # set when this file failed

class BatchPredictResponse(BaseModel):
    model_version: str
    results: list[ImageResult]  

class KeyRequest(BaseModel):
    name: str = "default"
