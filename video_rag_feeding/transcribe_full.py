import json
import logging
import sys
from pathlib import Path

def main(audio_path: str, output_path: str, device: str = "cpu"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("FullTranscriber")

    import numpy as np
    import librosa
    from transformers import pipeline

    logger.info(f"Loading ASR pipeline (Cohere) on {device}...")
    pipe = pipeline(
        task="automatic-speech-recognition",
        model="CohereLabs/cohere-transcribe-03-2026",
        device=device,
    )
    
    logger.info(f"Loading optimal audio matrix {audio_path}...")
    audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    logger.info("Transcribing optimal audio stream natively via manual chunking...")
    chunk_length_s = 20.0
    chunk_samples = int(sr * chunk_length_s)
    
    out_payload = []
    
    for i in range(0, len(audio_array), chunk_samples):
        chunk_data = audio_array[i:i+chunk_samples]
        
        # Omit nearly empty silent artifacts at the end
        if len(chunk_data) < sr * 0.5:
            continue
            
        start_time = float(i / sr)
        end_time = float((i + len(chunk_data)) / sr)
        
        audio_input = {
            "array": np.asarray(chunk_data, dtype=np.float32),
            "sampling_rate": sr,
        }
        
        raw = pipe(audio_input)
        text = raw["text"] if isinstance(raw, dict) else str(raw)
        
        if text.strip():
            out_payload.append({
                "start": start_time,
                "end": end_time,
                "text": text.strip()
            })
            
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump({"language": "en", "text": " ".join([c["text"] for c in out_payload]), "chunks": out_payload}, fp, indent=2)
        
    logger.info(f"Saved {len(out_payload)} highly accurate manual subtitle chunks to {output_path}")

if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    main("outputs/optimal_audio.wav", "outputs/full_transcript.json", device)
