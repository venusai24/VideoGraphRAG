from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Protocol, Sequence


class ClipNormalizationError(ValueError):
    """Raised when clip metadata cannot be normalized safely."""


class MediaExtractionError(RuntimeError):
    """Raised when frames or audio cannot be extracted from media."""


class VisionValidationError(ValueError):
    """Raised when a vision model response cannot be validated."""


@dataclass(slots=True)
class ClipLocator:
    clip_id: str
    start_time_sec: float
    end_time_sec: float
    clip_path: Optional[str] = None
    source_video_path: Optional[str] = None
    clip_fps: Optional[float] = None
    start_frame_index: Optional[int] = None
    end_frame_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    field_sources: Dict[str, str] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_time_sec - self.start_time_sec)


@dataclass(slots=True)
class FrameReference:
    frame_index: int
    timestamp_sec: float
    relative_offset_sec: float
    image_path: Optional[str] = None
    image: Optional[bytes] = None
    scores: Dict[str, float] = field(default_factory=dict)
    entities: Sequence[Any] = field(default_factory=tuple)


@dataclass(slots=True)
class VisionClipInput:
    clip: ClipLocator
    sampled_frames: Sequence[FrameReference]
    prompt_context: Dict[str, Any] = field(default_factory=dict)
    clip_video_path: Optional[str] = None


@dataclass(slots=True)
class AudioClipInput:
    clip: ClipLocator
    audio_array: Sequence[float]
    sample_rate_hz: int
    channel_mode: str
    audio_source: str

    @property
    def duration_sec(self) -> float:
        if not self.sample_rate_hz:
            return 0.0
        return len(self.audio_array) / float(self.sample_rate_hz)


@dataclass(slots=True)
class VisionEntity:
    name: str
    category: Optional[str] = None
    attributes: Sequence[str] = field(default_factory=tuple)
    evidence_frame_offsets_sec: Sequence[float] = field(default_factory=tuple)
    confidence: Optional[float] = None


@dataclass(slots=True)
class ActionRecord:
    description: str
    subject: Optional[str] = None
    object: Optional[str] = None
    evidence_frame_offsets_sec: Sequence[float] = field(default_factory=tuple)
    confidence: Optional[float] = None


@dataclass(slots=True)
class VisionExtraction:
    summary: str
    scene_context: str
    entities: Sequence[VisionEntity] = field(default_factory=tuple)
    actions: Sequence[ActionRecord] = field(default_factory=tuple)
    uncertainties: Sequence[str] = field(default_factory=tuple)
    raw_response: Any = None
    validation_status: str = "validated"


@dataclass(slots=True)
class TranscriptExtraction:
    text: str
    start_time_sec: float
    end_time_sec: float
    language: Optional[str] = None
    raw_response: Any = None
    validation_status: str = "validated"


@dataclass(slots=True)
class ClipEnrichmentResult:
    clip: ClipLocator
    vision: Optional[VisionExtraction] = None
    audio: Optional[TranscriptExtraction] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return jsonable(
            {
                "clip": self.clip,
                "vision": self.vision,
                "audio": self.audio,
                "provenance": self.provenance,
                "errors": self.errors,
            }
        )


class ClipSource(Protocol):
    def iter_clips(self) -> Iterator[ClipLocator]:
        """Yield normalized clips in chronological order."""


class VisionModelClient(Protocol):
    model_name: str

    def infer(self, batch: Sequence[VisionClipInput]) -> Sequence[VisionExtraction]:
        """Run vision inference for a batch of prepared clips."""


class AsrModelClient(Protocol):
    model_name: str

    def infer(self, batch: Sequence[AudioClipInput]) -> Sequence[TranscriptExtraction]:
        """Run ASR inference for a batch of prepared audio clips."""


def jsonable(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return {k: jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    return value
