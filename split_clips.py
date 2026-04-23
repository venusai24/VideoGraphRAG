import subprocess
import os

input_file = "Obama.mp4"
output_dir = "/home/venu/IR/VideoGraphRAG"
clip_duration = 30  # seconds
start_time = 30     # start from 30 seconds onward

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get total duration using ffprobe
def get_duration(file):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)

total_duration = get_duration(input_file)

clip_index = 0
current_time = start_time

while current_time < total_duration:
    output_file = os.path.join(output_dir, f"clip_{clip_index:03d}.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(current_time),
        "-i", input_file,
        "-t", str(clip_duration),
        "-c", "copy",
        output_file
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Created: {output_file}")

    current_time += clip_duration
    clip_index += 1