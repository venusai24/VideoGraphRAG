import os
import json

def print_json_structure(data, current_depth=0):
    prefix = "  " * current_depth

    if isinstance(data, dict):
        print(f"{prefix}{{object}}")
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}  '{key}' ->")
                # Go deeper for nested objects or arrays
                print_json_structure(value, current_depth + 1)
            else:
                # Print just the data type for primitive values
                type_name = "null" if value is None else type(value).__name__
                print(f"{prefix}  '{key}': {type_name}")

    elif isinstance(data, list):
        if len(data) > 0:
            print(f"{prefix}[array] containing:")
            # Analyze ONLY the first item to show the structural blueprint
            print_json_structure(data[0], current_depth + 1)
        else:
            print(f"{prefix}[empty array]")
            
    else:
        type_name = "null" if data is None else type(data).__name__
        print(f"{prefix}{type_name}")


def analyze_json_file(filepath):
    print(f"\nFILE: {os.path.basename(filepath)}")
    print("-" * 50)

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        print_json_structure(data)

    except Exception as e:
        print(f"Error reading file: {e}")


def analyze_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            analyze_json_file(filepath)


if __name__ == "__main__":
    directory_path = "/home/venu/IR/VideoGraphRAG/outputs/clip_0"
    analyze_directory(directory_path)