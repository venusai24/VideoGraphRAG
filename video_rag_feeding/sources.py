from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Union

from .contracts import ClipLocator, ClipNormalizationError, ClipSource


def _get(raw: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(raw, dict) and key in raw:
            return raw[key]
        if hasattr(raw, key):
            return getattr(raw, key)
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_path(value: Any, base_path: Optional[Path]) -> Optional[str]:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute() and base_path is not None:
        path = base_path / path
    return str(path.resolve())


def normalize_clip_record(
    raw: Any,
    *,
    base_path: Optional[Union[str, Path]] = None,
    default_clip_fps: float = 12.0,
    fallback_source_video_path: Optional[str] = None,
    record_index: Optional[int] = None,
) -> ClipLocator:
    base = Path(base_path).resolve() if base_path is not None else None
    metadata: Dict[str, Any]
    if isinstance(raw, dict):
        metadata = dict(raw)
    elif hasattr(raw, "__dict__"):
        metadata = dict(vars(raw))
    else:
        metadata = {"raw_value": raw}

    field_sources: Dict[str, str] = {}

    clip_id = _get(raw, "clip_id", "id")
    if clip_id is not None:
        field_sources["clip_id"] = "provided"
    else:
        clip_id = _get(raw, "clip_index", "cluster_id")
        if clip_id is not None:
            clip_id = f"clip_{int(clip_id):04d}"
            field_sources["clip_id"] = "inferred:record-index"

    clip_path = _resolve_path(_get(raw, "clip_path", "path", "video_path", "media_path"), base)
    if clip_path:
        field_sources["clip_path"] = "provided"
        if clip_id is None:
            clip_id = Path(clip_path).stem
            field_sources["clip_id"] = "inferred:path-stem"

    source_video_path = _resolve_path(
        _get(raw, "source_video_path", "source_path", "source_media_path"),
        base,
    )
    if source_video_path:
        field_sources["source_video_path"] = "provided"
    elif fallback_source_video_path:
        source_video_path = str(Path(fallback_source_video_path).resolve())
        field_sources["source_video_path"] = "inferred:fallback"

    clip_fps = _coerce_float(_get(raw, "clip_fps", "fps", "frame_rate"))
    if clip_fps is not None:
        field_sources["clip_fps"] = "provided"
    else:
        clip_fps = default_clip_fps
        field_sources["clip_fps"] = "inferred:default"

    start_time_sec = _coerce_float(_get(raw, "start_time_sec", "start_sec", "start_time", "start"))
    end_time_sec = _coerce_float(_get(raw, "end_time_sec", "end_sec", "end_time", "end"))

    start_frame_index = _coerce_int(_get(raw, "start_frame_index", "frame_start", "start_frame"))
    end_frame_index = _coerce_int(_get(raw, "end_frame_index", "frame_end", "end_frame"))
    if start_frame_index is not None:
        field_sources["start_frame_index"] = "provided"
    if end_frame_index is not None:
        field_sources["end_frame_index"] = "provided"

    if start_time_sec is None and start_frame_index is not None and clip_fps:
        start_time_sec = start_frame_index / clip_fps
        field_sources["start_time_sec"] = "inferred:frame-index"
    elif start_time_sec is not None:
        field_sources["start_time_sec"] = "provided"

    if end_time_sec is None and end_frame_index is not None and clip_fps:
        end_time_sec = end_frame_index / clip_fps
        field_sources["end_time_sec"] = "inferred:frame-index"
    elif end_time_sec is not None:
        field_sources["end_time_sec"] = "provided"

    if clip_id is None:
        if record_index is not None:
            clip_id = f"clip_{record_index:04d}"
            field_sources["clip_id"] = "inferred:iter-index"
        else:
            raise ClipNormalizationError("Unable to infer clip_id from record")

    if start_time_sec is None or end_time_sec is None:
        raise ClipNormalizationError(
            f"Clip {clip_id} is missing start/end times and they could not be inferred"
        )
    if end_time_sec <= start_time_sec:
        raise ClipNormalizationError(
            f"Clip {clip_id} has non-positive duration: start={start_time_sec}, end={end_time_sec}"
        )
    if clip_path is None and source_video_path is None:
        raise ClipNormalizationError(
            f"Clip {clip_id} has no clip_path or source_video_path media handle"
        )

    recognized = {
        "clip_id",
        "id",
        "clip_index",
        "cluster_id",
        "clip_path",
        "path",
        "video_path",
        "media_path",
        "source_video_path",
        "source_path",
        "source_media_path",
        "clip_fps",
        "fps",
        "frame_rate",
        "start_time_sec",
        "start_sec",
        "start_time",
        "start",
        "end_time_sec",
        "end_sec",
        "end_time",
        "end",
        "start_frame_index",
        "frame_start",
        "start_frame",
        "end_frame_index",
        "frame_end",
        "end_frame",
    }
    passthrough = {k: v for k, v in metadata.items() if k not in recognized}
    passthrough["normalization"] = {"field_sources": field_sources}

    return ClipLocator(
        clip_id=str(clip_id),
        start_time_sec=float(start_time_sec),
        end_time_sec=float(end_time_sec),
        clip_path=clip_path,
        source_video_path=source_video_path,
        clip_fps=float(clip_fps) if clip_fps is not None else None,
        start_frame_index=start_frame_index,
        end_frame_index=end_frame_index,
        metadata=passthrough,
        field_sources=field_sources,
    )


class ManifestClipSource:
    """Load clip metadata from a preprocessing manifest."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        *,
        default_clip_fps: float = 12.0,
        source_video_path: Optional[str] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.default_clip_fps = default_clip_fps
        self.source_video_path = source_video_path

    def iter_clips(self) -> Iterator[ClipLocator]:
        payload = json.loads(self.manifest_path.read_text())
        if not isinstance(payload, Sequence):
            raise ClipNormalizationError(
                f"Manifest {self.manifest_path} did not contain a list of clip records"
            )
        base_path = self.manifest_path.parent
        for index, item in enumerate(payload):
            yield normalize_clip_record(
                item,
                base_path=base_path,
                default_clip_fps=self.default_clip_fps,
                fallback_source_video_path=self.source_video_path,
                record_index=index,
            )


class IterableClipSource:
    """Normalize clip records from arbitrary dict/object iterables."""

    def __init__(
        self,
        records: Iterable[Any],
        *,
        base_path: Optional[Union[str, Path]] = None,
        default_clip_fps: float = 12.0,
        source_video_path: Optional[str] = None,
    ) -> None:
        self.records = records
        self.base_path = base_path
        self.default_clip_fps = default_clip_fps
        self.source_video_path = source_video_path

    def iter_clips(self) -> Iterator[ClipLocator]:
        for index, item in enumerate(self.records):
            if isinstance(item, ClipLocator):
                yield item
                continue
            yield normalize_clip_record(
                item,
                base_path=self.base_path,
                default_clip_fps=self.default_clip_fps,
                fallback_source_video_path=self.source_video_path,
                record_index=index,
            )


def resolve_clip_source(
    source: Union[str, Path, Iterable[Any], ClipSource],
    *,
    default_clip_fps: float = 12.0,
    source_video_path: Optional[str] = None,
) -> ClipSource:
    if hasattr(source, "iter_clips"):
        return source  # type: ignore[return-value]
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            manifest = path / "clips.json"
        else:
            manifest = path
        return ManifestClipSource(
            manifest,
            default_clip_fps=default_clip_fps,
            source_video_path=source_video_path,
        )
    return IterableClipSource(
        source,
        default_clip_fps=default_clip_fps,
        source_video_path=source_video_path,
    )
