"use client";

export default function UploadZone() {
  return (
    <div>
      <p>Upload car images</p>
      <input type="file" accept="image/*" multiple />
    </div>
  );
}
