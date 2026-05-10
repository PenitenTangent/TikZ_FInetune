import json
import hashlib
from pathlib import Path
from typing import Set

def load_quarantine_manifest(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            quarantined = data.get("quarantined_adapters", [])
            return {item.get("sha256") for item in quarantined if item.get("sha256")}
    except (json.JSONDecodeError, OSError):
        return set()

def assert_not_quarantined(adapter_path: Path, allow: bool = False) -> None:
    if allow:
        return
        
    if not adapter_path.exists():
        return
        
    # Standard location
    manifest_path = Path("runs/quarantine_manifest.json")
    quarantined_hashes = load_quarantine_manifest(manifest_path)
    
    if not quarantined_hashes:
        return
        
    # Compute sha256
    hasher = hashlib.sha256()
    with adapter_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
            
    adapter_hash = hasher.hexdigest()
    
    if adapter_hash in quarantined_hashes:
        raise RuntimeError(
            f"Adapter at {adapter_path} is QUARANTINED! "
            f"(SHA256: {adapter_hash}). "
            "Refusing to load or resume from this adapter. "
            "Pass --allow-quarantined to override."
        )
