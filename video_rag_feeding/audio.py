from __future__ import annotations

import io
import math
import struct
import subprocess
import wave
from array import array
from pathlib import Path
from typing import Iterable, List, Sequence

from .contracts import AudioClipInput, ClipLocator, MediaExtractionError


def _resolve_audio_media(clip: ClipLocator) -> tuple[str, float, float]:
    if clip.clip_path:
        return clip.clip_path, 0.0, clip.duration_sec
    if clip.source_video_path:
        return clip.source_video_path, clip.start_time_sec, clip.duration_sec
    raise MediaExtractionError(f"Clip {clip.clip_id} has no media path for audio extraction")


def read_wav_samples(wav_bytes: bytes) -> tuple[List[float], int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())

    if sample_width != 2:
        raise MediaExtractionError(f"Expected 16-bit audio, received sample width {sample_width}")
    samples = array("h")
    samples.frombytes(frames)
    if channels > 1:
        mono: List[float] = []
        for index in range(0, len(samples), channels):
            mono.append(sum(samples[index : index + channels]) / (channels * 32768.0))
        return mono, sample_rate
    return [sample / 32768.0 for sample in samples], sample_rate


def apply_noise_gate(
    samples: Sequence[float],
    sample_rate_hz: int,
    *,
    window_ms: int = 30,
    energy_ratio: float = 0.15,
) -> List[float]:
    if not samples:
        return []
    window_size = max(1, int(sample_rate_hz * window_ms / 1000.0))
    energies: List[float] = []
    for start in range(0, len(samples), window_size):
        window = samples[start : start + window_size]
        if not window:
            continue
        energy = math.sqrt(sum(sample * sample for sample in window) / len(window))
        energies.append(energy)
    max_energy = max(energies, default=0.0)
    if max_energy <= 0.0:
        return [0.0 for _ in samples]
    threshold = max_energy * energy_ratio
    gated: List[float] = []
    for window_index, start in enumerate(range(0, len(samples), window_size)):
        window = list(samples[start : start + window_size])
        if energies[window_index] < threshold:
            gated.extend([0.0] * len(window))
        else:
            gated.extend(window)
    return gated


def extract_audio_samples(
    clip: ClipLocator,
    *,
    sample_rate_hz: int = 16000,
) -> tuple[List[float], int, str]:
    media_path, seek_start, duration_sec = _resolve_audio_media(clip)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{seek_start:.6f}",
        "-t",
        f"{duration_sec:.6f}",
        "-i",
        media_path,
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        "-f",
        "wav",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise MediaExtractionError(
            f"Failed to extract audio for clip {clip.clip_id}: {result.stderr.decode('utf-8', 'ignore').strip()}"
        )
    samples, actual_sample_rate = read_wav_samples(result.stdout)
    return samples, actual_sample_rate, media_path


def prepare_audio_input(
    clip: ClipLocator,
    *,
    sample_rate_hz: int = 16000,
    apply_vad: bool = True,
) -> AudioClipInput:
    samples, actual_sample_rate, media_path = extract_audio_samples(
        clip,
        sample_rate_hz=sample_rate_hz,
    )
    gated = apply_noise_gate(samples, actual_sample_rate) if apply_vad else list(samples)
    return AudioClipInput(
        clip=clip,
        audio_array=tuple(gated),
        sample_rate_hz=actual_sample_rate,
        channel_mode="mono",
        audio_source=media_path,
    )


def group_audio_batches(
    inputs: Iterable[AudioClipInput],
    *,
    max_batch_audio_sec: float = 20.0,
) -> List[List[AudioClipInput]]:
    batches: List[List[AudioClipInput]] = []
    current: List[AudioClipInput] = []
    current_sec = 0.0
    for item in inputs:
        duration = item.duration_sec
        if current and current_sec + duration > max_batch_audio_sec:
            batches.append(current)
            current = []
            current_sec = 0.0
        current.append(item)
        current_sec += duration
    if current:
        batches.append(current)
    return batches


def serialize_audio_to_wav_bytes(audio_input: AudioClipInput) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(audio_input.sample_rate_hz)
        pcm = array(
            "h",
            [
                max(-32768, min(32767, int(sample * 32767.0)))
                for sample in audio_input.audio_array
            ],
        )
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()
