# Deprecated – not used in final pipeline
import os
import subprocess

# Base directory
BASE_DIR = "/mnt/MIG_store/Datasets/blending/madhav/VRAG"
INPUT_FILES = [
    "How to Skip 10 Seconds on Youtube (Desktop⧸Laptop⧸Macbook) ？ [uxg6OrseLa8].mp4"
]

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, "input")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_video(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-y",  # overwrite if exists
        output_path
    ]

    try:
        print(f"\n--- Converting: {input_path} ---")
        subprocess.run(cmd, check=True)
        print(f"✅ Saved to: {output_path}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to convert: {input_path}")

def main():
    for filename in INPUT_FILES:
        input_path = os.path.join(BASE_DIR, filename)
        
        # Clean output name (optional but safer)
        output_filename = filename.replace(" ", "_")
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        if not os.path.exists(input_path):
            print(f"⚠️ File not found: {input_path}")
            continue

        convert_video(input_path, output_path)

if __name__ == "__main__":
    main()