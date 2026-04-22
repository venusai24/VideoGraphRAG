import os
import json

def explore_json(data, indent=0, max_depth=3):
    prefix = "  " * indent

    if indent > max_depth:
        print(f"{prefix}... (max depth reached)")
        return

    if isinstance(data, dict):
        print(f"{prefix}{{object}} with {len(data)} keys")
        for key, value in data.items():
            print(f"{prefix}  '{key}' → {type(value).__name__}")
            explore_json(value, indent + 2, max_depth)

    elif isinstance(data, list):
        print(f"{prefix}[array] with {len(data)} items")
        if len(data) > 0:
            print(f"{prefix}  sample item:")
            explore_json(data[0], indent + 2, max_depth)

    else:
        print(f"{prefix}{data} ({type(data).__name__})")


def analyze_json_file(filepath):
    print(f"\nFILE: {os.path.basename(filepath)}")
    print("-" * 50)

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        print(f"Top-level type: {type(data).__name__}")
        explore_json(data)

    except Exception as e:
        print(f"Error reading file: {e}")


def analyze_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            analyze_json_file(filepath)


if __name__ == "__main__":
    directory_path = "outputs/video_indexer_test"
    analyze_directory(directory_path)