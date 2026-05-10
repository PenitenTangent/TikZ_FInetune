import json
from pathlib import Path
from typing import Dict, Any

def append_run_record(record: Dict[str, Any], path: Path = Path("runs/run_registry.jsonl")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
