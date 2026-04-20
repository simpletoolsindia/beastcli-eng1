"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function Bubble({ message }) {
  const parsed = message.parsedContent;
  const isSelf = message.role === "assistant";
  const isTool = message.role === "tool";
  const palette = isTool
    ? "bg-shell-bubbleTool border-shell-line"
    : isSelf
      ? "bg-shell-bubbleSelf border-emerald-700/40 ml-auto"
      : "bg-shell-bubbleOther border-shell-line";

  return (
    <div
      className={`max-w-[85%] rounded-2xl border px-4 py-3 shadow-sm ${
        message.role === "system" ? "bg-shell-bubbleWarn border-amber-700/30" : palette
      }`}
    >
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-shell-muted">
        <span>{message.role}</span>
        {message.name ? (
          <span className="rounded-full bg-black/20 px-2 py-0.5 normal-case tracking-normal text-shell-text">
            {message.name}
          </span>
        ) : null}
        {message.tool_call_id ? (
          <span className="rounded-full bg-black/20 px-2 py-0.5 normal-case tracking-normal text-shell-text">
            {message.tool_call_id}
          </span>
        ) : null}
      </div>
      {parsed ? (
        <div className="space-y-3">
          {parsed.type ? <div className="text-sm font-medium text-emerald-200">{parsed.type}</div> : null}
          <pre className="overflow-x-auto whitespace-pre-wrap break-words text-sm leading-6 text-shell-text">
            {prettyJson(parsed)}
          </pre>
        </div>
      ) : (
        <p className="whitespace-pre-wrap text-sm leading-6 text-shell-text">{message.content}</p>
      )}
    </div>
  );
}

function SampleCard({ sample, active, href }) {
  return (
    <Link
      href={href}
      className={`block rounded-2xl border px-4 py-3 transition ${
        active
          ? "border-emerald-500 bg-emerald-500/10"
          : "border-shell-line bg-shell-panel hover:border-emerald-500/40 hover:bg-shell-soft"
      }`}
    >
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-xs uppercase tracking-[0.2em] text-shell-muted">#{sample.rowNumber}</span>
        <span className="rounded-full bg-black/20 px-2 py-1 text-xs text-shell-text">
          {sample.metadata.tool_name}
        </span>
      </div>
      <p className="max-h-12 overflow-hidden text-sm leading-6 text-shell-text">{sample.title}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        <span className="rounded-full bg-black/20 px-2 py-1 text-xs text-shell-muted">
          {sample.metadata.difficulty}
        </span>
        {sample.suggestions?.length ? (
          <span className="rounded-full bg-emerald-500/15 px-2 py-1 text-xs text-emerald-200">
            {sample.suggestions.length} notes
          </span>
        ) : null}
        {sample.issueFlags.length ? (
          <span className="rounded-full bg-amber-500/15 px-2 py-1 text-xs text-amber-200">
            {sample.issueFlags.length} flagged
          </span>
        ) : null}
      </div>
    </Link>
  );
}

export default function ReviewWorkspace({
  datasets,
  dataset,
  samples,
  selectedIndex,
  totalLoaded,
  totalRows
}) {
  const datasetList = Array.isArray(datasets) ? datasets : [];
  const sampleList = Array.isArray(samples) ? samples : [];
  const router = useRouter();
  const fileInputRef = useRef(null);
  const selectedSample = sampleList[selectedIndex] || sampleList[0];
  const [draft, setDraft] = useState("");
  const [notes, setNotes] = useState(selectedSample?.suggestions || []);
  const [uploading, setUploading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const reviewInfo = useMemo(() => {
    return {
      datasetKey: dataset.key,
      rowNumber: selectedSample?.rowNumber ?? 1
    };
  }, [dataset.key, selectedSample?.rowNumber]);

  useEffect(() => {
    setNotes(selectedSample?.suggestions || []);
    setDraft("");
    setError("");
  }, [selectedSample?.id]);

  async function addNote() {
    const value = draft.trim();
    if (!value) return;
    setSaving(true);
    setError("");

    try {
      const response = await fetch("/api/reviews", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          dataset: reviewInfo.datasetKey,
          row: reviewInfo.rowNumber,
          suggestion: value
        })
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Failed to save suggestion.");
      }

      setNotes(payload.suggestions || []);
      setDraft("");
      router.refresh();
    } catch (caught) {
      setError(caught.message);
    } finally {
      setSaving(false);
    }
  }

  async function uploadDataset(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/datasets", {
        method: "POST",
        body: formData
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Failed to upload dataset.");
      }

      router.push(`/?dataset=${payload.dataset.key}&sample=0`);
      router.refresh();
    } catch (caught) {
      setError(caught.message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  if (!selectedSample) {
    return <div className="p-10 text-shell-text">No samples found.</div>;
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-[1800px] flex-col gap-6 px-4 py-6 lg:px-6">
      <div className="rounded-[28px] border border-shell-line bg-shell-panel/90 p-5 shadow-phone backdrop-blur">
        <div className="flex flex-col gap-5 xl:flex-row">
          <aside className="xl:w-[360px]">
            <div className="mb-4 flex items-start justify-between gap-3">
              <div>
                <h1 className="text-2xl font-semibold text-shell-text">Dataset Chat Reviewer</h1>
                <p className="mt-1 text-sm text-shell-muted">
                  WhatsApp-style JSONL review with row-numbered suggestions saved into JSON files.
                </p>
              </div>
            </div>

            <div className="mb-4 rounded-2xl border border-shell-line bg-shell-soft p-4">
              <div className="mb-3 text-xs uppercase tracking-[0.2em] text-shell-muted">Add JSONL</div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".jsonl"
                onChange={uploadDataset}
                className="block w-full rounded-xl border border-shell-line bg-shell-panel px-3 py-2 text-sm text-shell-text file:mr-3 file:rounded-lg file:border-0 file:bg-shell-accent file:px-3 file:py-2 file:text-sm file:font-semibold file:text-[#062a19]"
              />
              <p className="mt-2 text-xs leading-5 text-shell-muted">
                Upload a new JSONL and the app will create a separate review JSON keyed by row number.
              </p>
              {uploading ? <p className="mt-2 text-xs text-emerald-200">Uploading dataset...</p> : null}
            </div>

            <div className="mb-4 grid grid-cols-1 gap-3">
              {datasetList.map((item) => (
                <Link
                  key={item.key}
                  href={`/?dataset=${item.key}&sample=0`}
                  className={`rounded-2xl border px-4 py-3 text-sm ${
                    item.key === dataset.key
                      ? "border-emerald-500 bg-emerald-500/10 text-shell-text"
                      : "border-shell-line bg-shell-soft text-shell-muted"
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="font-medium text-shell-text">{item.label}</div>
                    <div className="rounded-full bg-black/20 px-2 py-1 text-[10px] uppercase tracking-[0.2em]">
                      {item.kind}
                    </div>
                  </div>
                  <div className="mt-2 text-xs text-shell-muted">{item.file}</div>
                </Link>
              ))}
            </div>

            <div className="mb-4 rounded-2xl border border-shell-line bg-shell-soft p-4">
              <div className="text-xs uppercase tracking-[0.2em] text-shell-muted">Loaded</div>
              <div className="mt-2 text-3xl font-semibold text-shell-text">{totalLoaded}</div>
              <div className="mt-1 text-sm text-shell-muted">Showing first rows from `{dataset.file}`</div>
              <div className="mt-3 text-xs text-shell-muted">Total rows in file: {totalRows}</div>
              <div className="mt-1 text-xs text-shell-muted">Review JSON: `{dataset.reviewFile}`</div>
            </div>

            <div className="h-[60vh] space-y-3 overflow-y-auto pr-1">
              {sampleList.map((sample) => (
                <SampleCard
                  key={sample.id}
                  sample={sample}
                  active={sample.index === selectedSample.index}
                  href={`/?dataset=${dataset.key}&sample=${sample.index}`}
                />
              ))}
            </div>
          </aside>

          <section className="min-w-0 flex-1">
            <div className="grid gap-5 2xl:grid-cols-[minmax(0,1.15fr)_420px]">
              <div className="rounded-[28px] border border-shell-line bg-[#0f1a20]">
                <div className="flex items-center justify-between border-b border-shell-line px-5 py-4">
                  <div>
                    <div className="text-xs uppercase tracking-[0.2em] text-shell-muted">
                      Row #{selectedSample.rowNumber}
                    </div>
                    <h2 className="mt-1 text-lg font-semibold text-shell-text">
                      {selectedSample.metadata.tool_name}
                    </h2>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <span className="rounded-full bg-black/20 px-3 py-1 text-xs text-shell-text">
                      {selectedSample.metadata.difficulty}
                    </span>
                    <span className="rounded-full bg-black/20 px-3 py-1 text-xs text-shell-text">
                      {selectedSample.localization.tone}
                    </span>
                  </div>
                </div>

                <div className="h-[72vh] space-y-4 overflow-y-auto bg-dots px-4 py-5 [background-size:16px_16px] sm:px-6">
                  {selectedSample.messages.map((message) => (
                    <Bubble key={message.id} message={message} />
                  ))}
                </div>
              </div>

              <div className="space-y-5">
                <div className="rounded-[28px] border border-shell-line bg-shell-soft p-5">
                  <div className="mb-4">
                    <div className="text-xs uppercase tracking-[0.2em] text-shell-muted">Suggestions</div>
                    <h3 className="mt-1 text-lg font-semibold text-shell-text">
                      Saved by row number
                    </h3>
                    <p className="mt-1 text-sm text-shell-muted">
                      Add review notes here and they will be written into `{dataset.reviewFile}` under row {selectedSample.rowNumber}.
                    </p>
                  </div>

                  <div className="space-y-3">
                    <textarea
                      value={draft}
                      onChange={(event) => setDraft(event.target.value)}
                      placeholder="Example: Row 12 should resolve the current directory before listing config."
                      className="min-h-28 w-full rounded-2xl border border-shell-line bg-shell-panel px-4 py-3 text-sm text-shell-text outline-none ring-0 placeholder:text-shell-muted"
                    />
                    <button
                      type="button"
                      onClick={addNote}
                      disabled={saving}
                      className="w-full rounded-2xl bg-shell-accent px-4 py-3 text-sm font-semibold text-[#062a19] transition hover:brightness-110 disabled:opacity-60"
                    >
                      {saving ? "Saving suggestion..." : "Add Suggestion"}
                    </button>
                    {error ? <p className="text-sm text-rose-300">{error}</p> : null}
                  </div>

                  <div className="mt-5 space-y-3">
                    {notes.length ? (
                      notes.map((note, index) => (
                        <div key={`${index}-${note}`} className="rounded-2xl border border-shell-line bg-shell-panel px-4 py-3">
                          <div className="mb-1 text-xs uppercase tracking-[0.2em] text-shell-muted">
                            Suggestion {index + 1}
                          </div>
                          <p className="text-sm leading-6 text-shell-text">{note}</p>
                        </div>
                      ))
                    ) : (
                      <div className="rounded-2xl border border-dashed border-shell-line px-4 py-5 text-sm text-shell-muted">
                        No suggestions yet for this row.
                      </div>
                    )}
                  </div>
                </div>

                <div className="rounded-[28px] border border-shell-line bg-shell-soft p-5">
                  <div className="text-xs uppercase tracking-[0.2em] text-shell-muted">Quick flags</div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {selectedSample.issueFlags.length ? (
                      selectedSample.issueFlags.map((flag) => (
                        <span
                          key={flag}
                          className="rounded-full bg-amber-500/15 px-3 py-1 text-xs text-amber-200"
                        >
                          {flag}
                        </span>
                      ))
                    ) : (
                      <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs text-emerald-200">
                        No automatic flags
                      </span>
                    )}
                  </div>

                  <div className="mt-5 space-y-3">
                    <div className="rounded-2xl bg-shell-panel p-4">
                      <div className="text-xs uppercase tracking-[0.2em] text-shell-muted">Metadata</div>
                      <pre className="mt-2 overflow-x-auto text-xs leading-6 text-shell-text">
                        {prettyJson(selectedSample.metadata)}
                      </pre>
                    </div>

                    <div className="rounded-2xl bg-shell-panel p-4">
                      <div className="text-xs uppercase tracking-[0.2em] text-shell-muted">Raw sample JSON</div>
                      <pre className="mt-2 max-h-[24rem] overflow-auto text-xs leading-6 text-shell-text">
                        {prettyJson(selectedSample.raw)}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
