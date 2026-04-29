import test from "node:test";
import assert from "node:assert/strict";

import { buildApiUrl, buildPlayableClipUrl, pickInitialSelectedClip } from "../lib/portal-state.js";

test("pickInitialSelectedClip chooses the first clip with playback support", () => {
  const selected = pickInitialSelectedClip([
    { clip_id: "clip-1", clip_path: null },
    { clip_id: "clip-2", clip_path: "/tmp/outputs/video.mp4" },
    { clip_id: "clip-3", clip_path: "/tmp/outputs/video-2.mp4" }
  ]);

  assert.equal(selected?.clip_id, "clip-2");
});

test("buildPlayableClipUrl uses clip_path to build the backend media route", () => {
  const playableUrl = buildPlayableClipUrl(
    "/tmp/outputs/abc123/clips/clip_0001.mp4",
    "http://localhost:8000"
  );
  assert.equal(
    playableUrl,
    "http://localhost:8000/media?clip_path=%2Ftmp%2Foutputs%2Fabc123%2Fclips%2Fclip_0001.mp4"
  );
});

test("buildApiUrl prefixes backend base for relative endpoints", () => {
  const apiUrl = buildApiUrl("/query/answer", "http://localhost:8000");
  assert.equal(apiUrl, "http://localhost:8000/query/answer");
});
