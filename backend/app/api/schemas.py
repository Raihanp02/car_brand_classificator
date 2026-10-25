"""Request/response schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel


class KeyRequest(BaseModel):
    name: str = "default"
