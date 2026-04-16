from __future__ import annotations

from typing import List, Sequence

from ..contracts import AudioClipInput, TranscriptExtraction


class TransformersAsrClient:
    """
    Optional Hugging Face ASR adapter for local Cohere Transcribe inference.

    Imports heavy ML dependencies lazily so the rest of the package remains usable
    on machines that only need orchestration and media preparation.
    """

    def __init__(
        self,
        *,
        model_name: str = "CohereLabs/cohere-transcribe-03-2026",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        import numpy as np
        from transformers import pipeline

        device = self.device if self.device is not None else "cpu"
        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.model_name,
            device=device,
        )
        self._np = np
        return self._pipeline

    def infer(self, batch: Sequence[AudioClipInput]) -> Sequence[TranscriptExtraction]:
        pipe = self._load_pipeline()
        outputs: List[TranscriptExtraction] = []
        for item in batch:
            audio_input = {
                "array": self._np.asarray(item.audio_array, dtype=self._np.float32),
                "sampling_rate": item.sample_rate_hz,
            }
            raw = pipe(audio_input)
            text = raw["text"] if isinstance(raw, dict) else str(raw)
            outputs.append(
                TranscriptExtraction(
                    text=text.strip(),
                    start_time_sec=item.clip.start_time_sec,
                    end_time_sec=item.clip.end_time_sec,
                    language=raw.get("language") if isinstance(raw, dict) else None,
                    raw_response=raw,
                )
            )
        return outputs
