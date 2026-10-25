"""Operator-only API-key issuance."""

from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException

from app.api.schemas import KeyRequest
from app.auth.store import create_key

router = APIRouter()


@router.post("/admin/keys")
def admin_create_key(payload: KeyRequest, x_admin_token: str | None = Header(default=None)):
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=403, detail="forbidden: operator only")
    raw = create_key(payload.name)
    return {"name": payload.name, "api_key": raw}
