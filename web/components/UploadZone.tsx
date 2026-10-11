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

/** Generate 3 procedural sample-car images client-side (no binaries to commit). */
async function makeSampleFiles(): Promise<File[]> {
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
      const body = (await res.json()) as { results: BatchItem[] };
      setResults(body.results);
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
    const files = await makeSampleFiles();
    await submit(files);
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
          {results.map((item) => (
            <ResultCard
              key={item.filename}
              item={item}
              previewUrl={previews.find((p) => p.file.name === item.filename)?.url}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}
