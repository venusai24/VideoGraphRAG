# Optimized for Apple M1 Pro (15GB RAM, Metal/MPS)
# -------------------------------------------------------------
# HARDWARE CONSTRAINTS:
# - Max total memory usage: ~12GB (to avoid swap death)
# - Run Audio and Vision strictly sequentially!
# -------------------------------------------------------------

.PHONY: all asr vlm clean

all: asr vlm
	@echo "End-to-End Pipeline Completed."

clean:
	rm -rf outputs/frames outputs/clips/frames outputs/enrichment*.jsonl

asr:
	@echo "=== STARTING ASR PIPELINE (Cohere-Transcribe) ==="
	@echo "Running purely on CPU/MPS to process audio. Free memory available."
	# Example command to run ASR stage sequentially
	# HF pipeline automatically leverages MPS if requested.
	python -c "from video_rag_feeding.orchestrator import run_feeding_pipeline; from video_rag_feeding.adapters.huggingface_asr import TransformersAsrClient; asr = TransformersAsrClient(device='mps'); run_feeding_pipeline(clip_source='outputs/clips/clips.json', vision_client=None, asr_client=asr, output_path='outputs/enrichment_audio.jsonl', workspace_dir='outputs/workspace_audio')"
	@echo "=== ASR COMPLETE ==="

vlm_server:
	@echo "=== STARTING LOCAL QWEN SERVER (MLX) ==="
	@echo "We use mlx_vlm.server for metal acceleration and context management."
	# User requested context window of upwards of 16k or more.
	./vgent/bin/python -m mlx_vlm.server --model MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bit --port 8080 --max-tokens 16384 &

vlm:
	@echo "=== STARTING VLM PIPELINE ==="
	@echo "Sending API requests to local mlx_vlm server on port 8080"
	./vgent/bin/python -c "from video_rag_feeding.orchestrator import run_feeding_pipeline; from video_rag_feeding.adapters.openai_compatible import OpenAICompatibleVisionClient; vlm = OpenAICompatibleVisionClient(endpoint_url='http://localhost:8080/v1/chat/completions', model_name='MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bit'); run_feeding_pipeline(clip_source='outputs/clips/clips.json', vision_client=vlm, asr_client=None, output_path='outputs/enrichment_vision.jsonl', workspace_dir='outputs/workspace_vision', vision_batch_size=2)"
	@echo "=== VLM COMPLETE ==="
