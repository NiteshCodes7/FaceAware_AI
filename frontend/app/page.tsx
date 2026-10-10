"use client";

import { useState, type DragEvent } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const CONTROLS = [
  { key: "skin_strength", label: "Skin", initial: 50 },
  { key: "eye_strength", label: "Eyes", initial: 40 },
  { key: "iris_strength", label: "Iris", initial: 60 },
  { key: "eye_bag_strength", label: "Eye Bags", initial: 50 },
  { key: "teeth_strength", label: "Teeth", initial: 70 },
  { key: "glasses_strength", label: "Glasses", initial: 80 },
  { key: "wrinkle_strength", label: "Wrinkles", initial: 40 },
] as const;

type Key = (typeof CONTROLS)[number]["key"];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [enhancedUrl, setEnhancedUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [values, setValues] = useState<Record<Key, number>>(
    () =>
      Object.fromEntries(CONTROLS.map((c) => [c.key, c.initial])) as Record<
        Key,
        number
      >
  );

  function loadFile(f: File | undefined) {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    if (enhancedUrl) URL.revokeObjectURL(enhancedUrl);
    setFile(f);
    setOriginalUrl(URL.createObjectURL(f));
    setEnhancedUrl(null);
    setError(null);
  }

  function onDrop(e: DragEvent<HTMLLabelElement>) {
    e.preventDefault();
    loadFile(e.dataTransfer.files?.[0]);
  }

  async function apply() {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const form = new FormData();
      form.append("image", file);
      for (const c of CONTROLS) {
        form.append(c.key, String(values[c.key] / 100));
      }

      const res = await fetch(`${API_URL}/enhance`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(
          typeof body?.detail === "string"
            ? body.detail
            : `Request failed (${res.status})`
        );
      }

      const blob = await res.blob();
      if (enhancedUrl) URL.revokeObjectURL(enhancedUrl);
      setEnhancedUrl(URL.createObjectURL(blob));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="mx-auto max-w-5xl px-4 py-10">
        <header className="mb-8 text-center">
          <h1 className="text-3xl font-bold tracking-tight">FaceAware AI</h1>
          <p className="mt-1 text-neutral-400">
            Region-Aware Facial Enhancement
          </p>
        </header>

        {/* Image panels */}
        <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Panel title="Original">
            <label
              onDrop={onDrop}
              onDragOver={(e) => e.preventDefault()}
              className="flex h-full w-full cursor-pointer items-center justify-center"
            >
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => loadFile(e.target.files?.[0])}
              />
              {originalUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={originalUrl}
                  alt="Original"
                  className="h-full w-full object-contain"
                />
              ) : (
                <span className="px-6 text-center text-neutral-500">
                  Click or drop an image here
                </span>
              )}
            </label>
          </Panel>

          <Panel title="Enhanced">
            {loading ? (
              <div className="flex h-full items-center justify-center gap-3 text-neutral-400">
                <span className="h-5 w-5 animate-spin rounded-full border-2 border-neutral-600 border-t-indigo-400" />
                Processing...
              </div>
            ) : enhancedUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={enhancedUrl}
                alt="Enhanced"
                className="h-full w-full object-contain"
              />
            ) : (
              <div className="flex h-full items-center justify-center text-neutral-500">
                Result appears here
              </div>
            )}
          </Panel>
        </section>

        {/* Sliders */}
        <section className="mt-6 space-y-4 rounded-xl border border-neutral-800 bg-neutral-900 p-6">
          {CONTROLS.map((c) => (
            <div key={c.key} className="flex items-center gap-4">
              <label
                htmlFor={c.key}
                className="w-24 shrink-0 text-sm font-medium text-neutral-300"
              >
                {c.label}
              </label>
              <input
                id={c.key}
                type="range"
                min={0}
                max={100}
                value={values[c.key]}
                onChange={(e) =>
                  setValues((v) => ({ ...v, [c.key]: Number(e.target.value) }))
                }
                className="h-2 flex-1 cursor-pointer accent-indigo-500"
              />
              <span className="w-12 text-right text-sm tabular-nums text-neutral-400">
                {values[c.key]}%
              </span>
            </div>
          ))}

          {error && (
            <p className="rounded-md bg-red-950 px-3 py-2 text-sm text-red-300">
              {error}
            </p>
          )}

          <div className="flex flex-col items-center gap-3 pt-2">
            <button
              onClick={apply}
              disabled={!file || loading}
              className="rounded-lg bg-indigo-600 px-8 py-3 font-semibold transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? "Processing..." : "Apply Enhancements"}
            </button>

            {enhancedUrl && !loading && (
              <a
                href={enhancedUrl}
                download="enhanced.jpg"
                className="text-sm text-indigo-400 hover:underline"
              >
                Download enhanced image
              </a>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

function Panel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900">
      <div className="border-b border-neutral-800 px-4 py-2 text-sm font-medium text-neutral-300">
        {title}
      </div>
      <div className="aspect-square bg-neutral-950">{children}</div>
    </div>
  );
}