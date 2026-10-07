# Car Classifier — Database ERD (v1)

> Source of truth for tables is `docs/superpowers/specs/2026-10-07-car-classifier-design.md §6b`.
> SQLite v1 (`backend/data/keys.db`). Postgres migration keeps the same 3 auth functions.

```mermaid
erDiagram
    api_keys ||--o{ usage_logs : "one key → many logs"
    api_keys {
        INTEGER id PK
        TEXT key_hash UK
        TEXT name
        TEXT created_at
        INTEGER revoked
        INTEGER rate_limit_per_day
    }
    usage_logs {
        INTEGER id PK
        INTEGER key_id FK
        TEXT ts
        TEXT endpoint
        TEXT top1_label
        REAL confidence
        INTEGER latency_ms
        INTEGER status_code
        TEXT ip
    }
```

## File stores (not tables)

- `models/index.faiss` — vectors (N, D), row i ↔ entry i in `index_meta.json`.
- `models/index_meta.json` — `{version, backbone, method, dim, entries[{row, class, view, source_path}]}`.
- `metadata.csv` — `{filepath, class, view, split, sha1, width, height, corrupt}` (see spec §5b).

## Post-v1 (Stripe phase, not built now)

```
users ||--o{ subscriptions : has
subscriptions ||--o{ api_keys : issues
```
Existing two tables unchanged; `api_keys` gains nullable `subscription_id FK`.
