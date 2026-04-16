from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .audio import group_audio_batches, prepare_audio_input
from .contracts import (
    AudioClipInput,
    ClipEnrichmentResult,
    ClipLocator,
    ClipSource,
    TranscriptExtraction,
    VisionClipInput,
    VisionExtraction,
    jsonable,
)
from .sources import resolve_clip_source
from .vision import PROMPT_VERSION, prepare_vision_input


@dataclass
class _PendingResult:
    clip: ClipLocator
    vision_input: Optional[VisionClipInput] = None
    audio_input: Optional[AudioClipInput] = None
    vision: Optional[VisionExtraction] = None
    audio: Optional[TranscriptExtraction] = None
    errors: Dict[str, str] = field(default_factory=dict)
    written: bool = False


class PipelineOrchestrator:
    def __init__(
        self,
        *,
        clip_source: ClipSource | Iterable[object] | str | Path,
        vision_client,
        asr_client,
        output_path: str | Path,
        workspace_dir: str | Path,
        vision_batch_size: int = 2,
        audio_batch_seconds: float = 20.0,
        resume: bool = True,
        vision_preparer: Callable[[ClipLocator, str | Path], VisionClipInput] = prepare_vision_input,
        audio_preparer: Callable[[ClipLocator], AudioClipInput] = prepare_audio_input,
    ) -> None:
        self.clip_source = resolve_clip_source(clip_source)
        self.vision_client = vision_client
        self.asr_client = asr_client
        self.output_path = Path(output_path).resolve()
        self.workspace_dir = Path(workspace_dir).resolve()
        self.vision_batch_size = max(1, vision_batch_size)
        self.audio_batch_seconds = max(0.1, audio_batch_seconds)
        self.resume = resume
        self.vision_preparer = vision_preparer
        self.audio_preparer = audio_preparer

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[ClipEnrichmentResult]:
        completed_ids = self._load_completed_ids() if self.resume else set()
        pending: Dict[str, _PendingResult] = {}
        written_results: List[ClipEnrichmentResult] = []
        queued_vision: List[VisionClipInput] = []
        queued_audio: List[AudioClipInput] = []

        for clip in self.clip_source.iter_clips():
            if clip.clip_id in completed_ids:
                continue
            state = _PendingResult(clip=clip)
            pending[clip.clip_id] = state

            try:
                state.vision_input = self.vision_preparer(clip, self.workspace_dir)
                queued_vision.append(state.vision_input)
            except Exception as exc:
                state.errors["vision_preparation"] = str(exc)

            try:
                state.audio_input = self.audio_preparer(clip)
                queued_audio.append(state.audio_input)
            except Exception as exc:
                state.errors["audio_preparation"] = str(exc)

            if len(queued_vision) >= self.vision_batch_size:
                self._flush_vision_batch(queued_vision, pending)
                queued_vision = []

            if sum(item.duration_sec for item in queued_audio) >= self.audio_batch_seconds:
                self._flush_audio_batches(queued_audio, pending)
                queued_audio = []

            written_results.extend(self._write_ready_results(pending))

        if queued_vision:
            self._flush_vision_batch(queued_vision, pending)
        if queued_audio:
            self._flush_audio_batches(queued_audio, pending)

        written_results.extend(self._write_ready_results(pending, force=True))
        return written_results

    def _flush_vision_batch(
        self,
        batch: Sequence[VisionClipInput],
        pending: Dict[str, _PendingResult],
    ) -> None:
        try:
            outputs = self.vision_client.infer(batch)
            if len(outputs) != len(batch):
                raise ValueError(
                    f"Vision client returned {len(outputs)} outputs for {len(batch)} inputs"
                )
            for item, output in zip(batch, outputs):
                pending[item.clip.clip_id].vision = output
        except Exception as exc:
            for item in batch:
                pending[item.clip.clip_id].errors["vision_inference"] = str(exc)

    def _flush_audio_batches(
        self,
        queued_audio: Sequence[AudioClipInput],
        pending: Dict[str, _PendingResult],
    ) -> None:
        for batch in group_audio_batches(queued_audio, max_batch_audio_sec=self.audio_batch_seconds):
            try:
                outputs = self.asr_client.infer(batch)
                if len(outputs) != len(batch):
                    raise ValueError(
                        f"ASR client returned {len(outputs)} outputs for {len(batch)} inputs"
                    )
                for item, output in zip(batch, outputs):
                    pending[item.clip.clip_id].audio = output
            except Exception as exc:
                for item in batch:
                    pending[item.clip.clip_id].errors["audio_inference"] = str(exc)

    def _build_result(self, state: _PendingResult) -> ClipEnrichmentResult:
        provenance = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "vision": {
                "model_name": getattr(self.vision_client, "model_name", self.vision_client.__class__.__name__),
                "prompt_version": PROMPT_VERSION,
                "sampled_frame_indices": (
                    [frame.frame_index for frame in state.vision_input.sampled_frames]
                    if state.vision_input
                    else []
                ),
                "sampled_frame_offsets_sec": (
                    [frame.relative_offset_sec for frame in state.vision_input.sampled_frames]
                    if state.vision_input
                    else []
                ),
                "clip_fps": state.clip.clip_fps,
                "validation_status": state.vision.validation_status if state.vision else "not_run",
            },
            "audio": {
                "model_name": getattr(self.asr_client, "model_name", self.asr_client.__class__.__name__),
                "audio_source": state.audio_input.audio_source if state.audio_input else None,
                "sample_rate_hz": state.audio_input.sample_rate_hz if state.audio_input else None,
                "validation_status": state.audio.validation_status if state.audio else "not_run",
            },
        }
        return ClipEnrichmentResult(
            clip=state.clip,
            vision=state.vision,
            audio=state.audio,
            provenance=provenance,
            errors=dict(state.errors),
        )

    def _write_ready_results(
        self,
        pending: Dict[str, _PendingResult],
        *,
        force: bool = False,
    ) -> List[ClipEnrichmentResult]:
        written: List[ClipEnrichmentResult] = []
        for clip_id, state in list(pending.items()):
            if state.written:
                continue
            vision_done = state.vision is not None or "vision_preparation" in state.errors or "vision_inference" in state.errors
            audio_done = state.audio is not None or "audio_preparation" in state.errors or "audio_inference" in state.errors
            if not force and not (vision_done and audio_done):
                continue
            result = self._build_result(state)
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(jsonable(result.to_json_dict())) + "\n")
            state.written = True
            written.append(result)
            pending.pop(clip_id, None)
        return written

    def _load_completed_ids(self) -> set[str]:
        if not self.output_path.exists():
            return set()
        completed = set()
        for line in self.output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            clip = payload.get("clip") or {}
            clip_id = clip.get("clip_id")
            if clip_id:
                completed.add(str(clip_id))
        return completed


def run_feeding_pipeline(
    *,
    clip_source,
    vision_client,
    asr_client,
    output_path: str | Path,
    workspace_dir: str | Path,
    vision_batch_size: int = 2,
    audio_batch_seconds: float = 20.0,
    resume: bool = True,
) -> List[ClipEnrichmentResult]:
    orchestrator = PipelineOrchestrator(
        clip_source=clip_source,
        vision_client=vision_client,
        asr_client=asr_client,
        output_path=output_path,
        workspace_dir=workspace_dir,
        vision_batch_size=vision_batch_size,
        audio_batch_seconds=audio_batch_seconds,
        resume=resume,
    )
    return orchestrator.run()
