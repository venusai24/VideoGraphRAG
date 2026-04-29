"use client";

import { useDeferredValue, useEffect, useMemo, useState, useTransition } from "react";

import {
  buildApiUrl,
  buildPlayableClipUrl,
  formatTimestampRange,
  pickInitialSelectedClip
} from "../lib/portal-state";
import { VideoPlayer } from "./video-player";

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

function normalizeErrorPayload(errorPayload) {
  if (!errorPayload) {
    return { message: "Unknown API error." };
  }

  if (typeof errorPayload === "string") {
    return { message: errorPayload };
  }

  return {
    stage: errorPayload.stage,
    message: errorPayload.error || errorPayload.detail || "Request failed.",
    timings: errorPayload.timings || null,
    results: errorPayload.results || null
  };
}

function TimingStat({ label, value }) {
  return (
    <div className="timing-stat">
      <span>{label}</span>
      <strong>{typeof value === "number" ? `${value.toFixed(2)} ms` : "n/a"}</strong>
    </div>
  );
}

function ResultCard({ clip, isActive, onSelect, apiBase, rank }) {
  const playableUrl = useMemo(
    () => buildPlayableClipUrl(clip.clip_path || clip.clip_url || null, apiBase),
    [apiBase, clip.clip_path, clip.clip_url]
  );

  return (
    <article className={`result-card${isActive ? " result-card-active" : ""}`}>
      <div className="result-card-header">
        <div>
          <p className="eyebrow">Rank {rank}</p>
          <h3>{clip.clip_id}</h3>
        </div>
        <button className="select-button" onClick={() => onSelect(clip)} type="button">
          {isActive ? "Selected" : playableUrl ? "Play clip" : "Inspect"}
        </button>
      </div>
      <div className="result-metrics">
        <span>Score {Number(clip.score ?? 0).toFixed(4)}</span>
        <span>{formatTimestampRange(clip.timestamp)}</span>
      </div>
      <p className="result-summary">{clip.summary || "No summary available for this clip."}</p>
      <div className="entity-row">
        {clip.entities?.length ? (
          clip.entities.map((entity) => (
            <span className="entity-pill" key={`${clip.clip_id}-${entity}`}>
              {entity}
            </span>
          ))
        ) : (
          <span className="entity-pill entity-pill-muted">No entities</span>
        )}
      </div>
    </article>
  );
}

export function QueryPortal() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("answer");
  const [submittedMode, setSubmittedMode] = useState("answer");
  const [topK, setTopK] = useState(10);
  const [responsePayload, setResponsePayload] = useState(null);
  const [selectedClip, setSelectedClip] = useState(null);
  const [errorState, setErrorState] = useState(null);
  const [showRawJson, setShowRawJson] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isPending, startTransition] = useTransition();
  const deferredPayload = useDeferredValue(responsePayload);

  useEffect(() => {
    if (!responsePayload?.results?.length) {
      setSelectedClip(null);
      return;
    }

    const stillPresent = responsePayload.results.find((clip) => clip.clip_id === selectedClip?.clip_id);
    if (!stillPresent) {
      setSelectedClip(pickInitialSelectedClip(responsePayload.results));
    }
  }, [responsePayload, selectedClip?.clip_id]);

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery || isSubmitting) {
      return;
    }

    setIsSubmitting(true);
    setErrorState(null);

    const endpoint = mode === "answer" ? "/query/answer" : "/query/retrieve";
    setSubmittedMode(mode);

    try {
      const request = await fetch(buildApiUrl(endpoint, DEFAULT_API_BASE), {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        cache: "no-store",
        body: JSON.stringify({
          query: trimmedQuery,
          top_k: topK
        })
      });

      const payload = await request.json().catch(() => null);

      if (!request.ok) {
        const normalized = normalizeErrorPayload(payload?.detail || payload);
        setErrorState(normalized);

        if (normalized.results?.length) {
          startTransition(() => {
            setResponsePayload({
              query: trimmedQuery,
              timings: normalized.timings,
              results: normalized.results
            });
            setSelectedClip(pickInitialSelectedClip(normalized.results));
          });
        }
        return;
      }

      startTransition(() => {
        setResponsePayload(payload);
        setSelectedClip(pickInitialSelectedClip(payload?.results || []));
      });
    } catch (error) {
      setErrorState(normalizeErrorPayload(error instanceof Error ? error.message : "Network request failed."));
    } finally {
      setIsSubmitting(false);
    }
  }

  const timings = responsePayload?.timings || errorState?.timings || {};
  const results = responsePayload?.results || [];
  const selectedPlayableClip = selectedClip || pickInitialSelectedClip(results);

  return (
    <main className="portal-page">
      <section className="hero">
        <p className="hero-kicker">VideoGraphRAG Live Portal</p>
        <h1>Query the pipeline directly, inspect ranked clips, and play the retrieved evidence.</h1>
        <p className="hero-copy">
          This UI stays thin: it forwards your query to the live backend, shows the ranked graph results,
          exposes pipeline latency, and loads the selected clip into a real video player.
        </p>
      </section>

      <section className="portal-grid">
        <aside className="panel panel-query">
          <div className="panel-header">
            <p className="eyebrow">Panel 1</p>
            <h2>Query Input</h2>
          </div>
          <form className="query-form" onSubmit={handleSubmit}>
            <label className="field-label" htmlFor="query-input">
              Query
            </label>
            <textarea
              id="query-input"
              className="query-input"
              placeholder="Ask an entity, temporal, or multi-hop question..."
              rows={5}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />

            <div className="mode-toggle">
              <button
                className={mode === "answer" ? "mode-button mode-button-active" : "mode-button"}
                onClick={() => setMode("answer")}
                type="button"
              >
                Full answer
              </button>
              <button
                className={mode === "retrieve" ? "mode-button mode-button-active" : "mode-button"}
                onClick={() => setMode("retrieve")}
                type="button"
              >
                Retrieval only
              </button>
            </div>

            <label className="field-label" htmlFor="topk-slider">
              Top-K: <strong>{topK}</strong>
            </label>
            <input
              id="topk-slider"
              className="slider"
              type="range"
              min="5"
              max="20"
              step="1"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />

            <button className="submit-button" disabled={!query.trim() || isSubmitting} type="submit">
              {isSubmitting ? "Running query..." : mode === "answer" ? "Run full pipeline" : "Run retrieval"}
            </button>
          </form>

          <div className="query-notes">
            <p className="eyebrow">Behavior</p>
            <p>{isPending ? "Rendering updated response..." : "Ready for live queries."}</p>
          </div>
        </aside>

        <section className="panel panel-results">
          <div className="panel-header">
            <p className="eyebrow">Panel 2</p>
            <h2>Results</h2>
          </div>

          {errorState ? (
            <div className="error-banner" role="alert">
              <strong>{errorState.stage ? `${errorState.stage} error` : "API error"}</strong>
              <p>{errorState.message}</p>
            </div>
          ) : null}

          {responsePayload?.answer ? (
            <article className="answer-card">
              <p className="eyebrow">Generated Answer</p>
              <h3>{responsePayload.answer.answer}</h3>
              <div className="confidence-bar-wrap">
                <div className="confidence-bar" style={{ width: `${Math.min(100, (Number(responsePayload.answer.confidence ?? 0) * 100))}%` }} />
                <span className="confidence-label">{(Number(responsePayload.answer.confidence ?? 0) * 100).toFixed(1)}% confidence</span>
              </div>
              <details className="reasoning-details">
                <summary>Show reasoning</summary>
                <p>{responsePayload.answer.reasoning}</p>
              </details>
              <div className="answer-meta">
                <span>
                  Citations: {responsePayload.answer.citations?.length ? responsePayload.answer.citations.join(", ") : "none"}
                </span>
              </div>
            </article>
          ) : null}

          <VideoPlayer clip={selectedPlayableClip} apiBase={DEFAULT_API_BASE} />

          <div className="results-toolbar">
            <p>
              {results.length ? `${results.length} retrieved result${results.length === 1 ? "" : "s"}` : "No results yet"}
            </p>
          </div>

          <div className="results-list">
            {results.length ? (
              results.map((clip, index) => (
                <ResultCard
                  apiBase={DEFAULT_API_BASE}
                  clip={clip}
                  isActive={clip.clip_id === selectedPlayableClip?.clip_id}
                  key={clip.clip_id}
                  onSelect={setSelectedClip}
                  rank={index + 1}
                />
              ))
            ) : (
              <div className="empty-card">
                <p className="eyebrow">Awaiting Query</p>
                <h3>No retrieval payload yet</h3>
                <p>Run a live retrieval or full-answer query to populate clips and playback.</p>
              </div>
            )}
          </div>
        </section>

        <aside className="panel panel-debug">
          <div className="panel-header">
            <p className="eyebrow">Panel 3</p>
            <h2>Pipeline Debug</h2>
          </div>

          <div className="timing-grid">
            <div className="timing-stat">
              <span>Mode</span>
              <strong>{submittedMode}</strong>
            </div>
            <TimingStat label="Decomposition" value={timings.decomposition} />
            <TimingStat label="Traversal" value={timings.traversal} />
            <TimingStat label="Ranking" value={timings.ranking} />
            <TimingStat label="Generation" value={timings.generation} />
            <TimingStat label="Total" value={timings.total} />
          </div>

          {responsePayload?.debug ? (
            <div className="debug-section">
              <p className="eyebrow">Debug Info</p>
              <div className="timing-grid">
                <div className="timing-stat">
                  <span>Decomp Provider</span>
                  <strong>{responsePayload.debug.provider_used || "—"}</strong>
                </div>
                <div className="timing-stat">
                  <span>Decomp Model</span>
                  <strong>{responsePayload.debug.model_used || "—"}</strong>
                </div>
                <div className="timing-stat">
                  <span>Gen Model</span>
                  <strong>{responsePayload.debug.gen_model_used || "—"}</strong>
                </div>
                <div className="timing-stat">
                  <span>Results</span>
                  <strong>{responsePayload.debug.num_results ?? "—"}</strong>
                </div>
                <div className="timing-stat">
                  <span>Media Used</span>
                  <strong>{responsePayload.debug.media_used ? "Yes" : "No"}</strong>
                </div>
              </div>
            </div>
          ) : null}

          <button className="raw-toggle" onClick={() => setShowRawJson((open) => !open)} type="button">
            {showRawJson ? "Hide raw JSON" : "Show raw JSON"}
          </button>

          {showRawJson ? (
            <pre className="raw-json">{JSON.stringify(deferredPayload || errorState || {}, null, 2)}</pre>
          ) : null}
        </aside>
      </section>
    </main>
  );
}
