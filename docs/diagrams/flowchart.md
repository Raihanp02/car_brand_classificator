# Car Classifier — User Activity Flowchart (v1)

> Demo visitor (no login) + API caller (manual key). No Stripe/accounts in v1.

```mermaid
flowchart TD
    A[Visitor opens Next.js demo] --> B{Has car photo?}
    B -- No --> S[Click Try sample cars]
    B -- Yes --> U[Click Upload / drag-drop 1-10 images]
    S --> P[Preview grid]
    U --> P
    P --> L{Rate limit OK? IP 20/day}
    L -- No --> R429[Show 429 + Retry-After]
    L -- Yes --> V{Images valid? size/type}
    V -- No --> R400[Inline per-image error]
    V -- Yes --> E[POST /v1/predict or /v1/predict/batch]
    E --> K[Backend: embed_batch + FAISS k=5 vote]
    K --> R[Result cards: top-1 + confidence bar + top-5]
    R --> M{Want API access?}
    M -- Yes --> C[Contact us → admin issues X-API-Key]
    M -- No --> D[Done]
    C --> G[Call API with X-API-Key header]
    G --> H{Key valid + quota?}
    H -- No --> R401[401 / 429]
    H -- Yes --> K
```

## Branches covered

1. Demo happy path: upload → results.
2. Sample path: no photo → bundled examples.
3. Guardrails: 400 corrupt, 413 too many/large, 429 rate-limited — all inline, batch returns per-item errors.
4. Conversion: demo → manual key → authenticated API (same classifier path).
