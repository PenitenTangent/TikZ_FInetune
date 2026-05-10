import hashlib
import json
from pathlib import Path
from typing import Optional, Any

def compute_sha256(filepath: str | Path, chunk_size: int = 8192) -> Optional[str]:
    """Compute the SHA256 hash of a file."""
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        return None
    sha256_hash = hashlib.sha256()
    with path.open("rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def write_adapter_load_manifest(
    manifest_path: str | Path,
    stage: str,
    base_model_id: str,
    adapter_path: Optional[str | Path],
    config_path: str | Path,
    lora_params: dict[str, Any],
    source_resume_adapter: Optional[str | Path],
    dataset_path: Optional[str | Path],
    pretokenized_cache_path: Optional[str | Path]
) -> None:
    """Write an adapter load manifest with cryptographic hashes."""
    manifest = {
        "stage": stage,
        "base_model_id": base_model_id,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_hash": compute_sha256(adapter_path) if adapter_path else None,
        "config_path": str(config_path),
        "config_hash": compute_sha256(config_path),
        "lora_params": lora_params,
        "source_resume_adapter": str(source_resume_adapter) if source_resume_adapter else None,
        "source_resume_adapter_hash": compute_sha256(source_resume_adapter) if source_resume_adapter else None,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "dataset_hash": compute_sha256(dataset_path) if dataset_path else None,
        "pretokenized_cache_path": str(pretokenized_cache_path) if pretokenized_cache_path else None,
        "pretokenized_cache_hash": compute_sha256(pretokenized_cache_path) if pretokenized_cache_path else None,
    }
    
    out_path = Path(manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
