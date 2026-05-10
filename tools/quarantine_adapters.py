import hashlib
import json
from pathlib import Path

def get_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    runs_dir = Path("runs")
    manifest_path = runs_dir / "quarantine_manifest.json"
    
    if manifest_path.exists():
        with manifest_path.open("r") as f:
            manifest = json.load(f)
    else:
        manifest = {"quarantined_adapters": []}
        
    quarantined_hashes = {item["sha256"] for item in manifest["quarantined_adapters"]}
    
    # Find stage 4 and 5 adapters
    for path in runs_dir.glob("**/adapters.safetensors"):
        rel_path = path.relative_to(runs_dir)
        if "stage4" in str(path) or "stage5" in str(path):
            h = get_sha256(path)
            if h not in quarantined_hashes:
                print(f"Quarantining {path} ({h})")
                manifest["quarantined_adapters"].append({
                    "path": str(path),
                    "sha256": h,
                    "reason": "Stage 4/5 adapter collapse (prompt contamination)"
                })
                quarantined_hashes.add(h)
                
    # Also check for individual safetensors files if any
    for path in runs_dir.glob("*.safetensors"):
        if "stage4" in path.name or "stage5" in path.name:
            h = get_sha256(path)
            if h not in quarantined_hashes:
                print(f"Quarantining {path} ({h})")
                manifest["quarantined_adapters"].append({
                    "path": str(path),
                    "sha256": h,
                    "reason": "Stage 4/5 adapter collapse (prompt contamination)"
                })
                quarantined_hashes.add(h)
                
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print("Quarantine manifest updated.")

if __name__ == "__main__":
    main()
