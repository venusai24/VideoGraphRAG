"""Post-preprocessing clip and audio feeding pipeline for VideoGraphRAG."""

from .contracts import (
    ActionRecord,
    AudioClipInput,
    AsrModelClient,
    ClipEnrichmentResult,
    ClipLocator,
    ClipSource,
    FrameReference,
    TranscriptExtraction,
    VisionClipInput,
    VisionEntity,
    VisionExtraction,
    VisionModelClient,
)
from .orchestrator import PipelineOrchestrator, run_feeding_pipeline
from .sources import ManifestClipSource, IterableClipSource, resolve_clip_source

__all__ = [
    "ActionRecord",
    "AudioClipInput",
    "AsrModelClient",
    "ClipEnrichmentResult",
    "ClipLocator",
    "ClipSource",
    "FrameReference",
    "IterableClipSource",
    "ManifestClipSource",
    "PipelineOrchestrator",
    "TranscriptExtraction",
    "VisionClipInput",
    "VisionEntity",
    "VisionExtraction",
    "VisionModelClient",
    "resolve_clip_source",
    "run_feeding_pipeline",
]
