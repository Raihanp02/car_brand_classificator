export type Top5Entry = { label: string; score: number };

export type BatchItem =
  | { filename: string; label: string; confidence: number; top5: Top5Entry[] }
  | { filename: string; error: string };

export function isErrorItem(item: BatchItem): item is { filename: string; error: string } {
  return "error" in item;
}

export default function ResultCard({
  item,
  previewUrl,
}: {
  item: BatchItem;
  previewUrl?: string;
}) {
  if (isErrorItem(item)) {
    return (
      <div data-testid="result-card" style={{ border: "1px solid #c00", padding: 12 }}>
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={previewUrl} alt={item.filename} width={160} />
        ) : null}
        <p>
          <strong>{item.filename}</strong>
        </p>
        <p role="alert" style={{ color: "#c00" }}>
          Error: {item.error}
        </p>
      </div>
    );
  }
  const pct = Math.round(item.confidence * 100);
  return (
    <div data-testid="result-card" style={{ border: "1px solid #ccc", padding: 12 }}>
      {previewUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={previewUrl} alt={item.filename} width={160} />
      ) : null}
      <p>
        <strong>{item.filename}</strong>
      </p>
      <p>
        {item.label} — {pct}%
      </p>
      <div
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        style={{ background: "#eee", height: 8 }}
      >
        <div style={{ background: "#06c", height: 8, width: `${pct}%` }} />
      </div>
      <details>
        <summary>Top 5</summary>
        <ol>
          {item.top5.map((t) => (
            <li key={t.label}>
              {t.label} — {Math.round(t.score * 100)}%
            </li>
          ))}
        </ol>
      </details>
    </div>
  );
}
