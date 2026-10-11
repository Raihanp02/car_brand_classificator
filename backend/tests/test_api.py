"""Task 3 API tests: health, auth, batch, classes, admin, rate limits."""

import io
import sqlite3

from fastapi.testclient import TestClient
from PIL import Image


def test_health_and_401():
    from app.api import app

    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.post("/v1/predict").status_code in (401, 422)  # no key / no file


# ---------------------------------------------------------------- fakes ---


def _png(color=(200, 30, 30), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


FAKE_META = {
    "method": "cls_proj256",
    "dim": 256,
    "entries": [
        {"class": "toyota_avanza", "view": "front"},
        {"class": "toyota_avanza", "view": "back"},
        {"class": "honda_brio", "view": "side"},
    ],
}

FAKE_HITS = [
    {"rank": 1, "class": "toyota_avanza", "score": 0.91},
    {"rank": 2, "class": "toyota_avanza", "score": 0.85},
    {"rank": 3, "class": "daihatsu_xenia", "score": 0.40},
    {"rank": 4, "class": "honda_brio", "score": 0.10},
    {"rank": 5, "class": "honda_brio", "score": 0.05},
]


class _FakeIndex:
    ntotal = 3


class _FakeEmbedder:
    def __init__(self):
        self.calls: list[int] = []

    def embed_batch(self, images):
        import torch

        self.calls.append(len(images))
        n = len(images)
        return torch.ones((n, 256), dtype=torch.float32) / 16.0


def _fake_search(emb, k=5, index=None, meta=None):
    try:
        n = emb.shape[0]
    except AttributeError:
        n = len(emb)
    return [[dict(h) for h in FAKE_HITS[:k]] for _ in range(n)]


def _mocked_client(monkeypatch, tmp_path):
    """TestClient with tmp keys.db, fake index/embedder/search (no weights)."""
    import app.api as api
    import app.auth as auth

    monkeypatch.setattr(auth, "DB_PATH", tmp_path / "keys.db")
    api.reset_state()
    fake_emb = _FakeEmbedder()
    monkeypatch.setattr(api, "get_index", lambda: (_FakeIndex(), FAKE_META))
    monkeypatch.setattr(api, "get_embedder", lambda: fake_emb)
    monkeypatch.setattr(api, "search_index", _fake_search)
    client = TestClient(api.app)
    client.fake_emb = fake_emb  # type: ignore[attr-defined]
    return client


def _key(client, name="t"):
    import app.auth as auth

    return auth.create_key(name)


# ------------------------------------------------------------- predict ---


def test_predict_ok_shape_and_logs_usage(monkeypatch, tmp_path):
    c = _mocked_client(monkeypatch, tmp_path)
    key = _key(c)
    r = c.post(
        "/v1/predict",
        files={"image": ("a.png", _png(), "image/png")},
        headers={"X-API-Key": key},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == "toyota_avanza"
    assert body["confidence"] == 0.91
    assert body["top5"][0] == {"label": "toyota_avanza", "score": 0.91}
    assert body["model_version"] == "registers-base_triplet_v1"
    assert isinstance(body["latency_ms"], int)
    assert c.fake_emb.calls == [1]  # single goes through batch-of-1

    import app.auth as auth

    con = sqlite3.connect(str(auth.DB_PATH))
    row = con.execute(
        "SELECT endpoint, status_code, top1_label FROM usage_logs"
    ).fetchone()
    con.close()
    assert row == ("/v1/predict", 200, "toyota_avanza")


def test_predict_keyless_demo_ok(monkeypatch, tmp_path):
    c = _mocked_client(monkeypatch, tmp_path)
    r = c.post("/v1/predict", files={"image": ("a.png", _png(), "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["label"] == "toyota_avanza"


def test_predict_bad_key_401(monkeypatch, tmp_path):
    c = _mocked_client(monkeypatch, tmp_path)
    r = c.post(
        "/v1/predict",
        files={"image": ("a.png", _png(), "image/png")},
        headers={"X-API-Key": "ck_bogus"},
    )
    assert r.status_code == 401
    assert "detail" in r.json()


def test_predict_revoked_key_401(monkeypatch, tmp_path):
    import app.auth as auth

    c = _mocked_client(monkeypatch, tmp_path)
    key = _key(c)
    assert auth.revoke_key(key) is True
    r = c.post(
        "/v1/predict",
        files={"image": ("a.png", _png(), "image/png")},
        headers={"X-API-Key": key},
    )
    assert r.status_code == 401


def test_predict_corrupt_400(monkeypatch, tmp_path):
    c = _mocked_client(monkeypatch, tmp_path)
    r = c.post(
        "/v1/predict",
        files={"image": ("a.png", b"not-an-image", "image/png")},
        headers={"X-API-Key": _key(c)},
    )
    assert r.status_code == 400
    assert "detail" in r.json()


def test_predict_oversize_413(monkeypatch, tmp_path):
    import app.api as api

    c = _mocked_client(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "MAX_BYTES", 10)
    r = c.post(
        "/v1/predict",
        files={"image": ("a.png", _png(), "image/png")},
        headers={"X-API-Key": _key(c)},
    )
    assert r.status_code == 413
    assert "detail" in r.json()


# --------------------------------------------------------------- batch ---


def test_batch_one_embed_call_per_item_ok(monkeypatch, tmp_path):
    c = _mocked_client(monkeypatch, tmp_path)
    files = [
        ("images", (f"{n}.png", _png((n * 40, 30, 30)), "image/png")) for n in range(3)
    ]
    r = c.post("/v1/predict/batch", files=files, headers={"X-API-Key": _key(c)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_version"] == "registers-base_triplet_v1"
    assert len(body["results"]) == 3
    assert all("label" in item for item in body["results"])
    assert c.fake_emb.calls == [3]  # exactly ONE embed_batch call


def test_batch_per_item_error_keeps_batch(monkeypatch, tmp_path):
    c = _mocked_client(monkeypatch, tmp_path)
    files = [
        ("images", ("good.png", _png(), "image/png")),
        ("images", ("bad.png", b"garbage-bytes", "image/png")),
    ]
    r = c.post("/v1/predict/batch", files=files, headers={"X-API-Key": _key(c)})
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert results[0]["filename"] == "good.png"
    assert results[0]["label"] == "toyota_avanza"
    assert results[1] == {"filename": "bad.png", "error": "bad.png: corrupt image"}
    assert c.fake_emb.calls == [1]


def test_batch_too_many_413(monkeypatch, tmp_path):
    c = _mocked_client(monkeypatch, tmp_path)
    files = [
        ("images", (f"{n}.png", _png(), "image/png")) for n in range(11)
    ]
    r = c.post("/v1/predict/batch", files=files, headers={"X-API-Key": _key(c)})
    assert r.status_code == 413
    assert "detail" in r.json()


def test_batch_empty_400(monkeypatch, tmp_path):
    c = _mocked_client(monkeypatch, tmp_path)
    r = c.post("/v1/predict/batch", files=[], headers={"X-API-Key": _key(c)})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_batch_oversize_is_per_item_error(monkeypatch, tmp_path):
    import app.api as api

    c = _mocked_client(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "MAX_BYTES", 10)
    files = [("images", ("big.png", _png(), "image/png"))]
    r = c.post("/v1/predict/batch", files=files, headers={"X-API-Key": _key(c)})
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["error"] == "file too large (>8MB)"


# -------------------------------------------------- classes / admin ---


def test_classes_counts(monkeypatch, tmp_path):
    c = _mocked_client(monkeypatch, tmp_path)
    r = c.get("/v1/classes")
    assert r.status_code == 200
    assert r.json() == {
        "classes": [
            {"name": "honda_brio", "count": 1},
            {"name": "toyota_avanza", "count": 2},
        ],
        "total": 2,
    }


def test_admin_keys_forbidden_without_token(monkeypatch, tmp_path):
    c = _mocked_client(monkeypatch, tmp_path)
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    r = c.post("/admin/keys", json={"name": "op"})
    assert r.status_code == 403
    assert "detail" in r.json()


def test_admin_keys_roundtrip(monkeypatch, tmp_path):
    c = _mocked_client(monkeypatch, tmp_path)
    monkeypatch.setenv("ADMIN_TOKEN", "secret-op")
    r = c.post(
        "/admin/keys", json={"name": "op"}, headers={"X-Admin-Token": "secret-op"}
    )
    assert r.status_code == 200, r.text
    raw = r.json()["api_key"]
    assert raw.startswith("ck_")
    # issued key authenticates (real sha256 path through tmp keys.db)
    r2 = c.post(
        "/v1/predict",
        files={"image": ("a.png", _png(), "image/png")},
        headers={"X-API-Key": raw},
    )
    assert r2.status_code == 200


def test_admin_keys_wrong_token_403(monkeypatch, tmp_path):
    c = _mocked_client(monkeypatch, tmp_path)
    monkeypatch.setenv("ADMIN_TOKEN", "secret-op")
    r = c.post("/admin/keys", json={"name": "op"}, headers={"X-Admin-Token": "nope"})
    assert r.status_code == 403


# ---------------------------------------------------------- rate limit ---


def test_keyless_demo_429_with_retry_after(monkeypatch, tmp_path):
    c = _mocked_client(monkeypatch, tmp_path)
    monkeypatch.setenv("DEMO_LIMIT_PER_DAY", "1")
    files = {"image": ("a.png", _png(), "image/png")}
    assert c.post("/v1/predict", files=files).status_code == 200
    r = c.post("/v1/predict", files=files)
    assert r.status_code == 429
    assert "detail" in r.json()
    assert "Retry-After" in r.headers
