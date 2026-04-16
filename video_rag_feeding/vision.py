from __future__ import annotations

import base64
import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from pydantic import BaseModel, Field, ValidationError

    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = object  # type: ignore[assignment]
    ValidationError = ValueError  # type: ignore[assignment]
    PYDANTIC_AVAILABLE = False

    def Field(*, default_factory=None, default=None):  # type: ignore[misc]
        if default_factory is not None:
            return default_factory()
        return default

from .contracts import (
    ActionRecord,
    ClipLocator,
    FrameReference,
    MediaExtractionError,
    VisionClipInput,
    VisionEntity,
    VisionExtraction,
    VisionValidationError,
)


PROMPT_VERSION = "qwen_clip_v1"


if PYDANTIC_AVAILABLE:

    class VisionEntityPayload(BaseModel):
        name: str
        category: Optional[str] = None
        attributes: List[str] = Field(default_factory=list)
        evidence_frame_offsets_sec: List[float] = Field(default_factory=list)
        confidence: Optional[float] = None


    class ActionPayload(BaseModel):
        description: str
        subject: Optional[str] = None
        object: Optional[str] = None
        evidence_frame_offsets_sec: List[float] = Field(default_factory=list)
        confidence: Optional[float] = None


    class VisionResponsePayload(BaseModel):
        clip_summary: str
        scene_context: str
        entities: List[VisionEntityPayload] = Field(default_factory=list)
        actions: List[ActionPayload] = Field(default_factory=list)
        uncertainties: List[str] = Field(default_factory=list)

else:

    @dataclass
    class VisionEntityPayload:
        name: str
        category: Optional[str] = None
        attributes: List[str] = field(default_factory=list)
        evidence_frame_offsets_sec: List[float] = field(default_factory=list)
        confidence: Optional[float] = None


    @dataclass
    class ActionPayload:
        description: str
        subject: Optional[str] = None
        object: Optional[str] = None
        evidence_frame_offsets_sec: List[float] = field(default_factory=list)
        confidence: Optional[float] = None


    @dataclass
    class VisionResponsePayload:
        clip_summary: str
        scene_context: str
        entities: List[VisionEntityPayload] = field(default_factory=list)
        actions: List[ActionPayload] = field(default_factory=list)
        uncertainties: List[str] = field(default_factory=list)

        @classmethod
        def model_validate(cls, payload: Dict[str, Any]) -> "VisionResponsePayload":
            if not isinstance(payload, dict):
                raise ValueError("Vision response payload must be a dictionary")
            clip_summary = payload.get("clip_summary")
            scene_context = payload.get("scene_context")
            if not isinstance(clip_summary, str) or not clip_summary.strip():
                raise ValueError("clip_summary must be a non-empty string")
            if not isinstance(scene_context, str) or not scene_context.strip():
                raise ValueError("scene_context must be a non-empty string")

            entities = []
            for raw_entity in payload.get("entities", []) or []:
                if not isinstance(raw_entity, dict) or not isinstance(raw_entity.get("name"), str):
                    raise ValueError("Each entity must be a dict with a string name")
                entities.append(
                    VisionEntityPayload(
                        name=raw_entity["name"],
                        category=raw_entity.get("category"),
                        attributes=[str(v) for v in raw_entity.get("attributes", []) or []],
                        evidence_frame_offsets_sec=[
                            float(v) for v in raw_entity.get("evidence_frame_offsets_sec", []) or []
                        ],
                        confidence=(
                            float(raw_entity["confidence"])
                            if raw_entity.get("confidence") is not None
                            else None
                        ),
                    )
                )

            actions = []
            for raw_action in payload.get("actions", []) or []:
                if not isinstance(raw_action, dict) or not isinstance(raw_action.get("description"), str):
                    raise ValueError("Each action must be a dict with a string description")
                actions.append(
                    ActionPayload(
                        description=raw_action["description"],
                        subject=raw_action.get("subject"),
                        object=raw_action.get("object"),
                        evidence_frame_offsets_sec=[
                            float(v) for v in raw_action.get("evidence_frame_offsets_sec", []) or []
                        ],
                        confidence=(
                            float(raw_action["confidence"])
                            if raw_action.get("confidence") is not None
                            else None
                        ),
                    )
                )

            uncertainties = [str(value) for value in payload.get("uncertainties", []) or []]
            return cls(
                clip_summary=clip_summary,
                scene_context=scene_context,
                entities=entities,
                actions=actions,
                uncertainties=uncertainties,
            )


def frame_budget_for_duration(duration_sec: float) -> int:
    if duration_sec < 3.0:
        return 4
    if duration_sec < 4.5:
        return 6
    return 8


def _uniform_indices(frame_count: int, sample_count: int) -> List[int]:
    if frame_count <= 0:
        return []
    if sample_count >= frame_count:
        return list(range(frame_count))
    if sample_count == 1:
        return [0]
    indices = []
    for position in range(sample_count):
        raw = round(position * (frame_count - 1) / float(sample_count - 1))
        indices.append(int(raw))
    deduped: List[int] = []
    for idx in indices:
        if not deduped or deduped[-1] != idx:
            deduped.append(idx)
    while len(deduped) < sample_count:
        candidate = min(frame_count - 1, deduped[-1] + 1)
        if candidate not in deduped:
            deduped.append(candidate)
        else:
            break
    return deduped[:sample_count]


def _normalize_salience(
    salience_scores: Optional[Mapping[int, float] | Sequence[float]],
    frame_count: int,
) -> Dict[int, float]:
    if salience_scores is None:
        return {}
    if isinstance(salience_scores, Mapping):
        return {int(k): float(v) for k, v in salience_scores.items()}
    return {idx: float(val) for idx, val in enumerate(salience_scores[:frame_count])}


def choose_sample_indices(
    frame_count: int,
    duration_sec: float,
    salience_scores: Optional[Mapping[int, float] | Sequence[float]] = None,
) -> List[int]:
    sample_count = min(frame_budget_for_duration(duration_sec), frame_count)
    base = _uniform_indices(frame_count, sample_count)
    if len(base) <= 2:
        return base

    salience = _normalize_salience(salience_scores, frame_count)
    if not salience:
        return base

    selected = [base[0]]
    for position in range(1, len(base) - 1):
        left = base[position - 1]
        center = base[position]
        right = base[position + 1]
        lower = max(left + 1, math.floor((left + center) / 2))
        upper = min(right - 1, math.ceil((center + right) / 2))
        candidates = [
            idx
            for idx in range(lower, upper + 1)
            if idx > selected[-1] and idx < base[-1]
        ]
        if not candidates:
            selected.append(center)
            continue
        replacement = max(
            candidates,
            key=lambda idx: (salience.get(idx, float("-inf")), -abs(idx - center)),
        )
        selected.append(replacement if replacement not in selected else center)
    selected.append(base[-1])
    return selected


def probe_video_metadata(video_path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    fps = _parse_ffprobe_rate(stream.get("avg_frame_rate"))
    nb_frames = stream.get("nb_frames")
    duration = stream.get("duration")
    return {
        "fps": fps,
        "frame_count": int(nb_frames) if str(nb_frames).isdigit() else None,
        "duration": float(duration) if duration is not None else None,
    }


def _parse_ffprobe_rate(rate: Any) -> Optional[float]:
    if not rate or rate == "0/0":
        return None
    if isinstance(rate, (int, float)):
        return float(rate)
    if "/" in str(rate):
        numerator, denominator = str(rate).split("/", 1)
        try:
            denominator_float = float(denominator)
            if denominator_float == 0:
                return None
            return float(numerator) / denominator_float
        except ValueError:
            return None
    try:
        return float(rate)
    except ValueError:
        return None


def _resolve_frame_count(clip: ClipLocator, clip_video_path: Optional[str]) -> int:
    candidates = [
        clip.metadata.get("frame_count"),
        clip.metadata.get("cluster_frame_count"),
    ]
    for candidate in candidates:
        try:
            if candidate is not None:
                return max(1, int(candidate))
        except (TypeError, ValueError):
            continue
    if clip.start_frame_index is not None and clip.end_frame_index is not None:
        return max(1, clip.end_frame_index - clip.start_frame_index + 1)
    if clip_video_path:
        metadata = probe_video_metadata(clip_video_path)
        if metadata.get("frame_count"):
            return int(metadata["frame_count"])
    fps = clip.clip_fps or 12.0
    return max(1, int(round(clip.duration_sec * fps)))


def _resolve_media_path(clip: ClipLocator) -> tuple[str, float]:
    if clip.clip_path:
        return clip.clip_path, 0.0
    if clip.source_video_path:
        return clip.source_video_path, clip.start_time_sec
    raise MediaExtractionError(f"Clip {clip.clip_id} has no media path")


def extract_sampled_frames(
    clip: ClipLocator,
    output_dir: str | Path,
    *,
    salience_scores: Optional[Mapping[int, float] | Sequence[float]] = None,
) -> List[FrameReference]:
    media_path, seek_base = _resolve_media_path(clip)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    clip_video_path = clip.clip_path or media_path
    frame_count = _resolve_frame_count(clip, clip_video_path)
    sample_indices = choose_sample_indices(frame_count, clip.duration_sec, salience_scores)
    fps = clip.clip_fps or 12.0

    sampled_frames: List[FrameReference] = []
    for ordinal, frame_index in enumerate(sample_indices):
        relative_offset_sec = min(clip.duration_sec, frame_index / fps)
        absolute_timestamp = clip.start_time_sec + relative_offset_sec
        frame_filename = f"{clip.clip_id}_f{ordinal:02d}_{frame_index:04d}.png"
        frame_path = output_path / frame_filename
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{seek_base + relative_offset_sec:.6f}",
            "-i",
            media_path,
            "-frames:v",
            "1",
            str(frame_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise MediaExtractionError(
                f"Failed to extract frame {frame_index} from clip {clip.clip_id}: {result.stderr.strip()}"
            )
        sampled_frames.append(
            FrameReference(
                frame_index=frame_index,
                timestamp_sec=absolute_timestamp,
                relative_offset_sec=relative_offset_sec,
                image_path=str(frame_path),
            )
        )
    return sampled_frames


def prepare_vision_input(
    clip: ClipLocator,
    workspace_dir: str | Path,
    *,
    salience_scores: Optional[Mapping[int, float] | Sequence[float]] = None,
) -> VisionClipInput:
    sampled_frames = extract_sampled_frames(
        clip,
        Path(workspace_dir) / "frames",
        salience_scores=salience_scores,
    )
    return VisionClipInput(
        clip=clip,
        sampled_frames=sampled_frames,
        prompt_context={
            "prompt_version": PROMPT_VERSION,
            "clip_fps": clip.clip_fps or 12.0,
            "frame_indices": [frame.frame_index for frame in sampled_frames],
            "frame_offsets_sec": [frame.relative_offset_sec for frame in sampled_frames],
        },
        clip_video_path=clip.clip_path,
    )


def build_qwen_prompt(clip_input: VisionClipInput) -> str:
    frame_lines = []
    for index, frame in enumerate(clip_input.sampled_frames):
        frame_lines.append(
            f"- frame_{index}: offset={frame.relative_offset_sec:.3f}s absolute_time={frame.timestamp_sec:.3f}s"
        )
    frame_block = "\n".join(frame_lines)
    return (
        "You are extracting graph-ready visual facts from a short video clip.\n"
        "Return JSON only. Do not wrap it in markdown.\n"
        "Use this schema exactly:\n"
        "{\n"
        '  "clip_summary": "string",\n'
        '  "scene_context": "string",\n'
        '  "entities": [{"name": "string", "category": "string|null", "attributes": ["string"], '
        '"evidence_frame_offsets_sec": [0.0], "confidence": 0.0}],\n'
        '  "actions": [{"description": "string", "subject": "string|null", "object": "string|null", '
        '"evidence_frame_offsets_sec": [0.0], "confidence": 0.0}],\n'
        '  "uncertainties": ["string"]\n'
        "}\n\n"
        f"Clip ID: {clip_input.clip.clip_id}\n"
        f"Clip start: {clip_input.clip.start_time_sec:.3f}s\n"
        f"Clip end: {clip_input.clip.end_time_sec:.3f}s\n"
        f"Clip fps: {clip_input.prompt_context.get('clip_fps', 12.0)}\n"
        "Sampled frames in chronological order:\n"
        f"{frame_block}\n\n"
        "Requirements:\n"
        "- Use stable, concrete entity names.\n"
        "- Keep actions observable and clip-bounded.\n"
        "- Use only listed frame offsets as evidence.\n"
        "- If uncertain, note it in uncertainties instead of inventing facts."
    )


def build_repair_prompt(raw_response: str) -> str:
    return (
        "Repair the following model output into valid JSON that matches the required schema exactly. "
        "Return JSON only and preserve meaning without adding new facts.\n\n"
        f"{raw_response}"
    )


def _extract_json_string(raw_text: str) -> str:
    stripped = raw_text.strip()
    code_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if code_match:
        return code_match.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise VisionValidationError("No JSON object found in vision model response")
    return stripped[start : end + 1]


def validate_vision_response(
    raw_response: str,
    *,
    repaired_response: Optional[str] = None,
) -> VisionExtraction:
    candidates = [raw_response]
    if repaired_response is not None:
        candidates.append(repaired_response)

    last_error: Optional[Exception] = None
    for index, candidate in enumerate(candidates):
        try:
            parsed = json.loads(_extract_json_string(candidate))
            payload = VisionResponsePayload.model_validate(parsed)
            return VisionExtraction(
                summary=payload.clip_summary,
                scene_context=payload.scene_context,
                entities=tuple(
                    VisionEntity(
                        name=item.name,
                        category=item.category,
                        attributes=tuple(item.attributes),
                        evidence_frame_offsets_sec=tuple(item.evidence_frame_offsets_sec),
                        confidence=item.confidence,
                    )
                    for item in payload.entities
                ),
                actions=tuple(
                    ActionRecord(
                        description=item.description,
                        subject=item.subject,
                        object=item.object,
                        evidence_frame_offsets_sec=tuple(item.evidence_frame_offsets_sec),
                        confidence=item.confidence,
                    )
                    for item in payload.actions
                ),
                uncertainties=tuple(payload.uncertainties),
                raw_response=candidate,
                validation_status="validated" if index == 0 else "validated_after_repair",
            )
        except (json.JSONDecodeError, ValidationError, VisionValidationError) as exc:
            last_error = exc
    raise VisionValidationError(str(last_error)) from last_error


def file_to_data_uri(path: str) -> str:
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"
