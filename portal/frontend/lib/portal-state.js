export function ensureTrailingSlash(value) {
  return value.endsWith("/") ? value : `${value}/`;
}

export function buildApiUrl(path, apiBase) {
  if (!path) {
    return ensureTrailingSlash(apiBase);
  }

  if (/^https?:\/\//.test(path)) {
    return path;
  }

  return new URL(path.replace(/^\//, ""), ensureTrailingSlash(apiBase)).toString();
}

export function buildPlayableClipUrl(clipUrl, apiBase) {
  if (!clipUrl) {
    return null;
  }

  if (/^https?:\/\//.test(clipUrl)) {
    return clipUrl;
  }

  if (clipUrl.startsWith("/media")) {
    return buildApiUrl(clipUrl, apiBase);
  }

  return buildApiUrl(`/media?clip_path=${encodeURIComponent(clipUrl)}`, apiBase);
}

export function pickInitialSelectedClip(results) {
  if (!Array.isArray(results)) {
    return null;
  }

  return results.find((result) => Boolean(result?.clip_path || result?.clip_url)) ?? results[0] ?? null;
}

export function formatTimestampRange(timestamp) {
  if (!timestamp) {
    return "Unknown range";
  }

  const start = Number(timestamp.start ?? 0);
  const end = Number(timestamp.end ?? 0);

  return `${start.toFixed(2)}s -> ${end.toFixed(2)}s`;
}
