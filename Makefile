# Optimized for Apple M1 Pro (15GB RAM, Metal/MPS)
# -------------------------------------------------------------
# HARDWARE CONSTRAINTS:
# - Max total memory usage: ~12GB (to avoid swap death)
# - Run Audio and Vision strictly sequentially!
# -------------------------------------------------------------

.PHONY: all asr vlm clean preprocess vlm_stop

# Default video path - can be overridden via command line
VIDEO_PATH ?= Project Hail Mary - Official Trailer.webm
OUTPUT_DIR ?= outputs/
HOMEBREW_PATH := /opt/homebrew/bin

all: clean preprocess asr vlm_server vlm vlm_stop
	@echo "End-to-End Pipeline Completed."

clean:
	rm -rf outputs/frames outputs/clips/ outputs/enrichment*.jsonl outputs/workspace* outputs/scores.json outputs/reconstructed.mp4

preprocess:
	@echo "=== STARTING PREPROCESSING PIPELINE ==="
	PATH=$(HOMEBREW_PATH):$(PATH) ./video_rag_preprocessing/venv/bin/python video_rag_preprocessing/run_pipeline.py --video_path "$(VIDEO_PATH)" --output_dir $(OUTPUT_DIR)
	@echo "=== PREPROCESSING COMPLETE ==="

asr:
	@echo "=== STARTING ASR PIPELINE (Cohere-Transcribe Full Audio) ==="
	@echo "Running on MPS/CPU to process the optimal raw audio stream."
	PATH="$(HOMEBREW_PATH):$(PATH)" ./vgent/bin/python video_rag_feeding/transcribe_full.py mps
	@echo "=== ASR COMPLETE ==="

vlm_server:
	@echo "=== STARTING LOCAL QWEN SERVER (MLX) ==="
	@if lsof -i :8080 > /dev/null; then echo "Server already running on 8080"; else \
		./vgent/bin/python -m mlx_vlm.server --model Qwen2.5-VL-7B-Instruct-8bit --port 8080 > vlm_server.log 2>&1 & \
		echo "Waiting for server to be ready..."; \
		for i in {1..30}; do \
			if curl -s http://localhost:8080/v1/models > /dev/null; then \
				echo "Server is UP!"; \
				break; \
			fi; \
			echo "Still waiting ($$i/30)..."; \
			sleep 2; \
		done; \
	fi

vlm:
	@echo "=== STARTING VLM PIPELINE === "
	@echo "Sending API requests to local mlx_vlm server on port 8080"
	PATH="$(HOMEBREW_PATH):$(PATH)" ./vgent/bin/python -c "from video_rag_feeding.orchestrator import run_feeding_pipeline; from video_rag_feeding.adapters.openai_compatible import OpenAICompatibleVisionClient; vlm = OpenAICompatibleVisionClient(endpoint_url='http://localhost:8080/v1/chat/completions', model_name='Qwen2.5-VL-7B-Instruct-8bit'); run_feeding_pipeline(clip_source='$(OUTPUT_DIR)clips/clips.json', vision_client=vlm, output_path='$(OUTPUT_DIR)enrichment_vision.jsonl', workspace_dir='$(OUTPUT_DIR)workspace_vision', vision_batch_size=2)"
	@echo "=== VLM COMPLETE ==="

vlm_stop:
	@echo "=== STOPPING VLM SERVER ==="
	@lsof -ti :8080 | xargs kill -9 || true
	@echo "=== SERVER STOPPED ==="
