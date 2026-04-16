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
	@echo "=== STARTING LOCAL QWEN SERVER (llama.cpp) ==="
	@echo "We use llama.cpp server for metal acceleration and context management."
	# -ngl 99 : offload all layers to Metal
	# -c 4096 : minimal context needed for 8 frames per clip
	# -np 2   : parallel batching slots
	llama-server -m models/qwen3.5-9b-vision.Q8_0.gguf -ngl 99 -c 4096 -np 2 --port 8080 &

vlm:
	@echo "=== STARTING VLM PIPELINE ==="
	@echo "Sending API requests to local llama-server on port 8080"
	python -c "from video_rag_feeding.orchestrator import run_feeding_pipeline; from video_rag_feeding.adapters.openai_compatible import OpenAICompatibleVisionClient; vlm = OpenAICompatibleVisionClient(endpoint_url='http://localhost:8080/v1/chat/completions', model_name='qwen-3.5-9b-vision'); run_feeding_pipeline(clip_source='outputs/clips/clips.json', vision_client=vlm, asr_client=None, output_path='outputs/enrichment_vision.jsonl', workspace_dir='outputs/workspace_vision', vision_batch_size=2)"
	@echo "=== VLM COMPLETE ==="
