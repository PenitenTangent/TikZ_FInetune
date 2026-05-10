import json
import re
import argparse
from pathlib import Path

def round_coordinates(text, precision=2):
    # Regex to find numbers that look like coordinates (e.g., 1.234567, -0.1234)
    # We look for numbers with more than 'precision' decimal places
    def repl(match):
        val = float(match.group(0))
        return f"{val:.{precision}f}".rstrip('0').rstrip('.')
    
    # Matches floats like 1.2345, -1.2345, but avoids matching things like version numbers if possible
    # We target numbers inside TikZ-like contexts if we want to be very specific, 
    # but a general float rounding in the assistant content is usually safe for TikZ.
    return re.sub(r"-?\d+\.\d{3,}", repl, text)

def process_file(input_path, output_path):
    with open(input_path, 'r') as f, open(output_path, 'w') as out:
        for line in f:
            data = json.loads(line)
            for msg in data.get("messages", []):
                if msg["role"] == "assistant":
                    content = msg["content"]
                    if isinstance(content, list):
                        for item in content:
                            if item["type"] == "text":
                                item["text"] = round_coordinates(item["text"])
                    elif isinstance(content, str):
                        msg["content"] = round_coordinates(content)
            out.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    process_file(args.input, args.output)
