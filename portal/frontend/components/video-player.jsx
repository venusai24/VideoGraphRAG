"use client";

import { useEffect, useMemo, useRef } from "react";

import { buildPlayableClipUrl, formatTimestampRange } from "../lib/portal-state";

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export function VideoPlayer({ clip, apiBase = DEFAULT_API_BASE }) {
  const videoRef = useRef(null);
  const playableUrl = useMemo(
    () => buildPlayableClipUrl(clip?.clip_path || clip?.clip_url || null, apiBase),
    [apiBase, clip?.clip_path, clip?.clip_url]
  );

  useEffect(() => {
    const node = videoRef.current;
    if (!node || !playableUrl || !clip?.timestamp) {
      return undefined;
    }

    const seekToClipStart = () => {
      const clipStart = Number(clip.timestamp?.start ?? 0);
      if (Number.isFinite(clipStart) && clipStart >= 0) {
        try {
          node.currentTime = clipStart;
        } catch (error) {
          console.warn("Unable to seek video preview", error);
        }
      }
    };

    if (node.readyState >= 1) {
      seekToClipStart();
    }

    node.addEventListener("loadedmetadata", seekToClipStart);
    return () => node.removeEventListener("loadedmetadata", seekToClipStart);
  }, [clip?.clip_id, clip?.timestamp, playableUrl]);

  if (!clip) {
    return (
      <div className="player-shell player-empty">
        <p className="eyebrow">Video Player</p>
        <h3>No clip selected</h3>
        <p>Select a retrieved clip to load it into the player.</p>
      </div>
    );
  }

  return (
    <div className="player-shell">
      <div className="player-heading">
        <div>
          <p className="eyebrow">Video Player</p>
          <h3>{clip.clip_id}</h3>
        </div>
        <div className="player-meta">
          <span>Score {Number(clip.score ?? 0).toFixed(4)}</span>
          <span>{formatTimestampRange(clip.timestamp)}</span>
        </div>
      </div>
      {playableUrl ? (
        <video
          key={clip.clip_id}
          ref={videoRef}
          className="video-frame"
          controls
          preload="metadata"
          src={playableUrl}
        />
      ) : (
        <div className="player-unavailable">
          <p>Playback unavailable for this result.</p>
          <p className="muted">The clip metadata exists, but no browser-safe media URL was returned.</p>
        </div>
      )}
      <div className="player-details">
        <span>{clip.timestamp?.video_id || "Unknown video"}</span>
        <span>{clip.summary || "No clip summary available."}</span>
      </div>
    </div>
  );
}
