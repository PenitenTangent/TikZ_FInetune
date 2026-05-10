#!/usr/bin/env python3
"""Quarantine one or more adapters by adding them to the manifest.

Usage:
  # Quarantine a specific adapter:
  python3 tools/quarantine_adapters.py --adapter runs/curriculum_stage1/final_adapter.safetensors \\
      --reason "post-training eval gate failed"

  # Legacy scan-mode (stage4/5 by name):
  python3 tools/quarantine_adapters.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "runs" / "quarantine_manifest.json"


def _get_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"quarantined_adapters": []}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def _quarantine(path: Path, reason: str, manifest: dict) -> bool:
    """Add adapter to quarantine manifest. Returns True if newly added."""
    if not path.exists():
        print(f"WARNING: adapter not found, cannot quarantine: {path}", file=sys.stderr)
        return False

    h = _get_sha256(path)
    existing_hashes = {item["sha256"] for item in manifest["quarantined_adapters"]}
    if h in existing_hashes:
        print(f"  Already quarantined: {path} ({h[:12]}…)")
        return False

    manifest["quarantined_adapters"].append({
        "path": str(path.resolve()),
        "sha256": h,
        "reason": reason,
        "quarantined_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"  ✓ Quarantined {path.name} ({h[:12]}…) — {reason}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Quarantine an adapter to prevent future use.")
    parser.add_argument("--adapter", help="Path to adapter .safetensors to quarantine")
    parser.add_argument("--reason", default="manually quarantined", help="Reason for quarantine")
    parser.add_argument(
        "--scan-stage45",
        action="store_true",
        default=False,
        help="Also scan runs/ for stage4/stage5 adapters (legacy mode)",
    )
    args = parser.parse_args()

    manifest = _load_manifest()
    changed = False

    # --- Targeted quarantine ---
    if args.adapter:
        adapter_path = Path(args.adapter).expanduser().resolve()
        if _quarantine(adapter_path, args.reason, manifest):
            changed = True

    # --- Legacy stage4/5 scan ---
    if args.scan_stage45 or not args.adapter:
        runs_dir = ROOT_DIR / "runs"
        for glob in ["**/adapters.safetensors", "*.safetensors"]:
            for path in runs_dir.glob(glob):
                if "stage4" in str(path) or "stage5" in str(path):
                    if _quarantine(path, "Stage 4/5 adapter collapse (prompt contamination)", manifest):
                        changed = True

    if changed:
        _save_manifest(manifest)
        print(f"Manifest updated: {MANIFEST_PATH}")
    else:
        print("No new adapters quarantined.")


if __name__ == "__main__":
    main()
