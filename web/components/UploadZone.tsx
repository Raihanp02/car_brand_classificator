"use client";

import { useRef, useState } from "react";
import ResultCard, { type BatchItem } from "./ResultCard";

const MAX_FILES = 10;
const MAX_SIDE_PX = 1024;

function apiBase(): string {
  return process.env.NEXT_PUBLIC_API_URL ?? "";
}

/** Downscale to max 1024px side via canvas; returns original when already small. */
async function downscale(file: File): Promise<File> {
  const bitmap = await createImageBitmap(file);
  try {
    const scale = Math.min(1, MAX_SIDE_PX / Math.max(bitmap.width, bitmap.height));
    if (scale === 1) return file;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return file;
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.9),
    );
    if (!blob) return file;
    return new File([blob], file.name, { type: "image/jpeg" });
  } finally {
    bitmap.close();
  }
}

/** Bundled sample-car fixtures served from web/public/samples/. */
const SAMPLE_URLS = [
  "/samples/sample-car-1.jpg",
  "/samples/sample-car-2.jpg",
  "/samples/sample-car-3.jpg",
];

/** Load the 3 bundled sample images via fetch() → File. */
async function loadSampleFiles(): Promise<File[]> {
  const out: File[] = [];
  for (let i = 0; i < SAMPLE_URLS.length; i++) {
    const res = await fetch(SAMPLE_URLS[i]);
    if (!res.ok) throw new Error(`Sample image ${i + 1} failed to load (${res.status})`);
    const blob = await res.blob();
    out.push(new File([blob], `sample-car-${i + 1}.jpg`, { type: blob.type || "image/jpeg" }));
  }
  return out;
}

/** Fallback only: 3 procedural canvas JPEGs when bundled samples can't load. */
async function makeFallbackSampleFiles(): Promise<File[]> {
  const colors = ["#b03a2e", "#2874a6", "#239b56"];
  const out: File[] = [];
  for (let i = 0; i < 3; i++) {
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = colors[i];
    ctx.fillRect(0, 0, 640, 480);
    ctx.fillStyle = "#fff";
    ctx.font = "48px sans-serif";
    ctx.fillText(`sample car ${i + 1}`, 180, 250);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.9),
    );
    if (blob) out.push(new File([blob], `sample-car-${i + 1}.jpg`, { type: "image/jpeg" }));
  }
  return out;
}

/** Raw backend payload shapes (batch contract may drift: top5 vs top_5). */
type RawTopEntry = { label?: unknown; score?: unknown };
type RawBatchItem = {
  filename?: unknown;
  label?: unknown;
  confidence?: unknown;
  top5?: unknown;
  top_5?: unknown;
  error?: unknown;
};

function clamp01(n: number): number {
  if (!Number.isFinite(n)) return 0;
  return Math.min(1, Math.max(0, n));
}

function toTopEntry(raw: unknown, fallbackLabel: string): { label: string; score: number } {
  const r = (raw ?? {}) as RawTopEntry;
  return {
    label: typeof r.label === "string" && r.label ? r.label : fallbackLabel,
    score: clamp01(typeof r.score === "number" ? r.score : NaN),
  };
}

/** Validate/normalize one batch item so a contract mismatch renders gracefully. */
function normalizeItem(raw: unknown, index: number): BatchItem {
  const r = (raw ?? {}) as RawBatchItem;
  const filename =
    typeof r.filename === "string" && r.filename ? r.filename : `image-${index + 1}`;
  if (typeof r.error === "string" && r.error) return { filename, error: r.error };
  const list = Array.isArray(r.top5) ? r.top5 : Array.isArray(r.top_5) ? r.top_5 : [];
  return {
    filename,
    label: typeof r.label === "string" && r.label ? r.label : "Unknown",
    confidence: clamp01(typeof r.confidence === "number" ? r.confidence : NaN),
    top5: list.map((t, i) => toTopEntry(t, `rank-${i + 1}`)),
  };
}

/** Validate/normalize the batch response body; null = unusable shape. */
function normalizeResults(body: unknown): BatchItem[] | null {
  const raw = body as { results?: unknown } | unknown[];
  const list = Array.isArray(raw) ? raw : raw && typeof raw === "object" ? raw.results : undefined;
  if (!Array.isArray(list)) return null;
  return list.map((item, i) => normalizeItem(item, i));
}

export default function UploadZone() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [previews, setPreviews] = useState<{ file: File; url: string }[]>([]);
  const [results, setResults] = useState<BatchItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  async function submit(files: File[]) {
    setError(null);
    setResults(null);
    if (files.length < 1) {
      setError("Select at least 1 image.");
      return;
    }
    if (files.length > MAX_FILES) {
      setError(`Too many images: max ${MAX_FILES} per batch.`);
      return;
    }
    const base = apiBase();
    if (!base) {
      setError("NEXT_PUBLIC_API_URL is not set — copy web/.env.example to web/.env.local.");
      return;
    }
    setLoading(true);
    try {
      const downsized = await Promise.all(files.map(downscale));
      setPreviews((old) => {
        old.forEach((p) => URL.revokeObjectURL(p.url));
        return downsized.map((file) => ({ file, url: URL.createObjectURL(file) }));
      });
      const form = new FormData();
      downsized.forEach((f) => form.append("images", f, f.name));
      const res = await fetch(`${base}/v1/predict/batch`, { method: "POST", body: form });
      if (!res.ok) {
        let detail = `Request failed (${res.status})`;
        try {
          const body = (await res.json()) as { detail?: string };
          if (body.detail) detail = body.detail;
        } catch {
          /* keep default */
        }
        if (res.status === 429) {
          const retryAfter = res.headers.get("Retry-After");
          detail += retryAfter ? ` Retry after ${retryAfter}s.` : " Try again later.";
        }
        setError(detail);
        return;
      }
      const body: unknown = await res.json();
      const items = normalizeResults(body);
      if (!items) {
        setError("Unexpected API response shape.");
        return;
      }
      setResults(items);
    } catch (e) {
      setError(e instanceof Error ? `Network error: ${e.message}` : "Network error");
    } finally {
      setLoading(false);
    }
  }

  function onInputFiles(list: FileList | null) {
    if (list) void submit(Array.from(list));
  }

  async function onSample() {
    try {
      await submit(await loadSampleFiles());
    } catch {
      // Bundled samples unreachable (e.g. offline preview): fall back to canvas.
      await submit(await makeFallbackSampleFiles());
    }
  }

  return (
    <div>
      <p>Upload car images</p>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          void submit(Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/")));
        }}
        style={{
          border: `2px dashed ${dragOver ? "#06c" : "#999"}`,
          padding: 24,
          marginBottom: 12,
        }}
      >
        Drag &amp; drop 1–10 car images here, or
        <div>
          <button type="button" onClick={() => inputRef.current?.click()} disabled={loading}>
            {loading ? "Classifying…" : "Choose images"}
          </button>{" "}
          <button type="button" onClick={() => void onSample()} disabled={loading}>
            Try sample cars
          </button>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => {
            onInputFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>
      {error ? (
        <p role="alert" data-testid="form-error">
          {error}
        </p>
      ) : null}
      {results ? (
        <div
          data-testid="results-grid"
          style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12 }}
        >
          {results.map((item, index) => (
            <ResultCard
              key={`${item.filename}-${index}`}
              item={item}
              previewUrl={previews[index]?.url}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}
