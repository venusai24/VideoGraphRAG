import os
import subprocess

PARENT_DIR = "/mnt/MIG_store/Datasets/blending/madhav/VRAG"

def convert_all():
    for filename in os.listdir(PARENT_DIR):
        input_path = os.path.join(PARENT_DIR, filename)

        # Only files (skip directories)
        if not os.path.isfile(input_path):
            continue

        # Only .mp4
        if not filename.lower().endswith(".mp4"):
            continue

        base_name = os.path.splitext(filename)[0]

        # Remove spaces + append _enc
        clean_name = base_name.replace(" ", "_") + "_enc.mp4"
        output_path = os.path.join(PARENT_DIR, clean_name)

        print(f"\n--- Converting: {filename} ---")
        print(f"→ Output: {clean_name}")

        cmd = [
            "ffmpeg",
            "-err_detect", "ignore_err",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-y",  # overwrite output if exists
            output_path
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Done: {clean_name}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed: {filename}")

if __name__ == "__main__":
    convert_all()