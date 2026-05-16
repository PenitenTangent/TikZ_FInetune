from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


def assign_example_index(record: MutableMapping[str, Any], example_index: int) -> MutableMapping[str, Any]:
    """Assign a row-stable training index at top level and in metadata."""

    if isinstance(example_index, bool):
        raise TypeError("example_index must be an integer, not bool")
    if example_index < 0:
        raise ValueError(f"example_index must be non-negative, got {example_index}")

    index = int(example_index)
    record["example_index"] = index

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        record["metadata"] = metadata
    metadata["example_index"] = index
    return record
