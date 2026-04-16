from __future__ import annotations

from typing import Callable, Sequence

from ..contracts import (
    AsrModelClient,
    AudioClipInput,
    TranscriptExtraction,
    VisionClipInput,
    VisionExtraction,
    VisionModelClient,
)


class CallableVisionClient(VisionModelClient):
    def __init__(
        self,
        handler: Callable[[Sequence[VisionClipInput]], Sequence[VisionExtraction]],
        *,
        model_name: str = "callable-vision-client",
    ) -> None:
        self._handler = handler
        self.model_name = model_name

    def infer(self, batch: Sequence[VisionClipInput]) -> Sequence[VisionExtraction]:
        return self._handler(batch)


class CallableAsrClient(AsrModelClient):
    def __init__(
        self,
        handler: Callable[[Sequence[AudioClipInput]], Sequence[TranscriptExtraction]],
        *,
        model_name: str = "callable-asr-client",
    ) -> None:
        self._handler = handler
        self.model_name = model_name

    def infer(self, batch: Sequence[AudioClipInput]) -> Sequence[TranscriptExtraction]:
        return self._handler(batch)
