from __future__ import annotations

import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

from video_rag_feeding.adapters.callable import CallableAsrClient, CallableVisionClient
from video_rag_feeding.audio import (
    apply_noise_gate,
    group_audio_batches,
    prepare_audio_input,
)
from video_rag_feeding.contracts import (
    AudioClipInput,
    ClipLocator,
    TranscriptExtraction,
    VisionClipInput,
    VisionExtraction,
)
from video_rag_feeding.orchestrator import PipelineOrchestrator
from video_rag_feeding.sources import IterableClipSource, ManifestClipSource
from video_rag_feeding.vision import (
    build_qwen_prompt,
    choose_sample_indices,
    validate_vision_response,
)


class ClipSourceTests(unittest.TestCase):
    def test_manifest_source_preserves_unknown_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            clip_path = tmp / "clip_0000.mp4"
            clip_path.write_bytes(b"placeholder")
            manifest = tmp / "clips.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "clip_index": 0,
                            "start_time_sec": 1.0,
                            "end_time_sec": 3.0,
                            "path": "clip_0000.mp4",
                            "score_profile": {"semantic_avg": 0.9},
                        }
                    ]
                )
            )

            clip = next(ManifestClipSource(manifest).iter_clips())
            self.assertEqual(clip.clip_id, "clip_0000")
            self.assertEqual(Path(clip.clip_path), clip_path.resolve())
            self.assertIn("score_profile", clip.metadata)
            self.assertEqual(clip.field_sources["clip_id"], "inferred:record-index")

    def test_iterable_source_infers_timestamps_from_frames(self):
        records = [
            {
                "id": "abc",
                "start_frame_index": 12,
                "end_frame_index": 35,
                "fps": 12,
                "source_video_path": "demo.mp4",
            }
        ]
        clip = next(IterableClipSource(records, base_path=".").iter_clips())
        self.assertEqual(clip.clip_id, "abc")
        self.assertAlmostEqual(clip.start_time_sec, 1.0)
        self.assertAlmostEqual(clip.end_time_sec, 35 / 12.0)
        self.assertEqual(clip.field_sources["end_time_sec"], "inferred:frame-index")


class VisionTests(unittest.TestCase):
    def test_frame_policy_uses_expected_counts_for_12fps_clips(self):
        self.assertEqual(len(choose_sample_indices(24, 2.0)), 4)
        self.assertEqual(len(choose_sample_indices(36, 3.0)), 6)
        self.assertEqual(len(choose_sample_indices(54, 4.5)), 8)
        self.assertEqual(len(choose_sample_indices(72, 6.0)), 8)

    def test_salience_substitution_preserves_order(self):
        salience = {8: 0.2, 15: 0.9, 25: 0.1, 33: 0.95, 40: 0.1}
        indices = choose_sample_indices(48, 4.0, salience_scores=salience)
        self.assertEqual(indices, sorted(indices))
        self.assertIn(15, indices)
        self.assertIn(33, indices)

    def test_prompt_and_validation_repair_flow(self):
        clip = ClipLocator(
            clip_id="clip_001",
            start_time_sec=0.0,
            end_time_sec=3.0,
            clip_path="/tmp/clip.mp4",
            clip_fps=12.0,
        )
        vision_input = VisionClipInput(
            clip=clip,
            sampled_frames=[],
            prompt_context={"clip_fps": 12.0},
        )
        prompt = build_qwen_prompt(vision_input)
        self.assertIn("Clip fps: 12.0", prompt)

        extraction = validate_vision_response(
            "not-json",
            repaired_response=json.dumps(
                {
                    "clip_summary": "A person waves.",
                    "scene_context": "Indoor room",
                    "entities": [{"name": "person", "category": "human"}],
                    "actions": [{"description": "waves", "subject": "person"}],
                    "uncertainties": [],
                }
            ),
        )
        self.assertEqual(extraction.summary, "A person waves.")
        self.assertEqual(extraction.validation_status, "validated_after_repair")


class AudioTests(unittest.TestCase):
    def test_noise_gate_zeros_low_energy_windows(self):
        samples = [0.0] * 320 + [0.7] * 320 + [0.01] * 320
        gated = apply_noise_gate(samples, 16000, window_ms=20, energy_ratio=0.2)
        self.assertTrue(all(value == 0.0 for value in gated[:320]))
        self.assertTrue(any(value != 0.0 for value in gated[320:640]))
        self.assertTrue(all(value == 0.0 for value in gated[640:]))

    def test_audio_extraction_respects_clip_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "tone.wav"
            self._write_tone(wav_path, duration_sec=1.0, sample_rate=16000)
            clip = ClipLocator(
                clip_id="clip_audio",
                start_time_sec=0.25,
                end_time_sec=0.75,
                source_video_path=str(wav_path),
                clip_fps=12.0,
            )
            prepared = prepare_audio_input(clip)
            self.assertAlmostEqual(prepared.duration_sec, 0.5, delta=0.05)
            self.assertEqual(prepared.sample_rate_hz, 16000)

    def test_audio_batching_uses_total_duration(self):
        clip = ClipLocator(
            clip_id="x",
            start_time_sec=0.0,
            end_time_sec=1.0,
            source_video_path="demo.wav",
            clip_fps=12.0,
        )
        inputs = [
            AudioClipInput(clip, [0.0] * 16000, 16000, "mono", "a"),
            AudioClipInput(clip, [0.0] * 16000, 16000, "mono", "b"),
            AudioClipInput(clip, [0.0] * 16000, 16000, "mono", "c"),
        ]
        batches = group_audio_batches(inputs, max_batch_audio_sec=2.0)
        self.assertEqual([len(batch) for batch in batches], [2, 1])

    def _write_tone(self, path: Path, *, duration_sec: float, sample_rate: int) -> None:
        frame_count = int(duration_sec * sample_rate)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            frames = bytearray()
            for index in range(frame_count):
                sample = int(16000 * math.sin(2.0 * math.pi * 220.0 * index / sample_rate))
                frames.extend(int(sample).to_bytes(2, byteorder="little", signed=True))
            handle.writeframes(bytes(frames))


class OrchestratorTests(unittest.TestCase):
    def test_orchestrator_writes_resume_safe_jsonl(self):
        clip_a = ClipLocator(
            clip_id="clip_a",
            start_time_sec=0.0,
            end_time_sec=2.0,
            clip_path="/tmp/clip_a.mp4",
            clip_fps=12.0,
        )
        clip_b = ClipLocator(
            clip_id="clip_b",
            start_time_sec=2.0,
            end_time_sec=4.0,
            clip_path="/tmp/clip_b.mp4",
            clip_fps=12.0,
        )

        def prepare_vision(clip, _workspace):
            return VisionClipInput(clip=clip, sampled_frames=[], prompt_context={})

        def prepare_audio(clip):
            return AudioClipInput(
                clip=clip,
                audio_array=[0.0] * 1600,
                sample_rate_hz=16000,
                channel_mode="mono",
                audio_source="memory",
            )

        vision_client = CallableVisionClient(
            lambda batch: [
                VisionExtraction(summary=item.clip.clip_id, scene_context="room")
                for item in batch
            ],
            model_name="vision-mock",
        )
        asr_client = CallableAsrClient(
            lambda batch: [
                TranscriptExtraction(
                    text=f"speech-{item.clip.clip_id}",
                    start_time_sec=item.clip.start_time_sec,
                    end_time_sec=item.clip.end_time_sec,
                    language="en",
                )
                for item in batch
            ],
            model_name="asr-mock",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "enrichment.jsonl"
            workspace = Path(tmpdir) / "workspace"
            source = IterableClipSource([clip_a, clip_b])
            orchestrator = PipelineOrchestrator(
                clip_source=source,
                vision_client=vision_client,
                asr_client=asr_client,
                output_path=output_path,
                workspace_dir=workspace,
                vision_preparer=prepare_vision,
                audio_preparer=prepare_audio,
            )
            first_run = orchestrator.run()
            self.assertEqual(len(first_run), 2)

            second_run = PipelineOrchestrator(
                clip_source=source,
                vision_client=vision_client,
                asr_client=asr_client,
                output_path=output_path,
                workspace_dir=workspace,
                vision_preparer=prepare_vision,
                audio_preparer=prepare_audio,
            ).run()
            self.assertEqual(second_run, [])

            lines = output_path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            payload = json.loads(lines[0])
            self.assertIn("provenance", payload)
            self.assertIn("clip", payload)


if __name__ == "__main__":
    unittest.main()
