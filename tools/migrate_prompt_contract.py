import json
import sys
from pathlib import Path
from tqdm import tqdm

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tikz_mlx.dataset import iter_jsonl, write_jsonl
from tikz_mlx.normalize import normalize_for_training_target
from tikz_mlx.prompting import build_generation_prompt

def migrate_record(record: dict) -> dict:
    messages = record.get("messages")
    if not messages or len(messages) < 2:
        return record
        
    # 1. Get description and metadata
    metadata = record.get("metadata", {})
    generation_mode = metadata.get("generation_mode")
    geometry_hints = metadata.get("geometry_hints")
    
    # Extract description from OLD prompt if possible, or use metadata
    user_content = messages[0].get("content")
    if isinstance(user_content, list):
        user_text = "".join(p.get("text", "") for p in user_content if p.get("type") == "text")
    else:
        user_text = user_content
        
    # We try to extract the description part (between "requirements:\n" and "\n\n[GEOMETRY HINTS]")
    import re
    desc_match = re.search(r"requirements:\n(.*?)\n\n\[GEOMETRY HINTS\]", user_text, re.DOTALL)
    if desc_match:
        description = desc_match.group(1).strip()
    else:
        # Fallback: if not found, we might have to rely on a generic or the whole text?
        # But our prompts were structured. Let's try another split.
        description = user_text.split("\n")[1] if len(user_text.split("\n")) > 1 else user_text
    
    # 2. Get assistant code and re-normalize to be body-only
    assistant_content = messages[1].get("content")
    if isinstance(assistant_content, list):
        assistant_text = "".join(p.get("text", "") for p in assistant_content if p.get("type") == "text")
    else:
        assistant_text = assistant_content
        
    # Strip markdown fence
    code = assistant_text.replace("```latex", "").replace("```", "").strip()
    
    # RE-NORMALIZE (this is the key fix)
    body = normalize_for_training_target(code)
    
    # 3. Rebuild prompt
    new_prompt = build_generation_prompt(
        description,
        generation_mode=generation_mode,
        geometry_hints=geometry_hints
    )
    
    # 4. Update record
    record["messages"] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": new_prompt}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": body + "\n```\n"}]
        }
    ]
    
    return record

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="JSONL files to migrate")
    args = parser.parse_args()
    
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"Skipping missing {f}")
            continue
            
        print(f"Migrating {f}...")
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in tqdm(handle):
                if line.strip():
                    records.append(migrate_record(json.loads(line)))
        
        # Overwrite with migrated records
        write_jsonl(path, records)
        print(f"Successfully migrated {len(records)} records in {f}")

if __name__ == "__main__":
    main()
