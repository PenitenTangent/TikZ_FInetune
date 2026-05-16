from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .bad_patterns import check_bad_patterns
from .prompting import PROMPT_CONTRACT_VERSION, build_generation_prompt, prompt_template_sha256
from .token_stats import command_counts, dominant_command_ratio, COORD_RE

SUBSTANTIVE_COMMANDS = {
    "\\draw",
    "\\node",
    "\\path",
    "\\fill",
    "\\filldraw",
    "\\coordinate",
    "\\addplot",
}

PREAMBLE_COMMANDS = {
    "\\documentclass",
    "\\usepackage",
    "\\usetikzlibrary",
    "\\PreviewEnvironment",
}

@dataclass
class SubstantiveFeatures:
    has_tikz_environment: bool
    valid_command_count: int
    coordinate_count: int
    label_count: int
    preamble_command_count: int
    bad_pattern_count: int
    dominant_command_ratio: float
    non_preamble_command_ratio: float
    substantive_score: float
    substantive_pass: bool

def substantive_features(text: str) -> dict[str, Any]:
    cmds = command_counts(text)
    
    valid_command_count = sum(cmds.get(cmd, 0) for cmd in SUBSTANTIVE_COMMANDS)
    preamble_command_count = sum(cmds.get(cmd, 0) for cmd in PREAMBLE_COMMANDS)
    
    total_cmds = sum(cmds.values())
    non_preamble_command_ratio = 1.0
    if total_cmds > 0:
        non_preamble_command_ratio = (total_cmds - preamble_command_count) / float(total_cmds)
        
    _, _, dom_ratio = dominant_command_ratio(text)
    
    has_tikz_environment = bool(re.search(r"\\begin\{(?:tikzpicture|tikz-cd|tikzcd|circuitikz|axis)\}", text, re.IGNORECASE))
    coordinate_count = len(COORD_RE.findall(text))
    
    # Simple heuristic for labels: node[...] {text}
    label_count = len(re.findall(r"\\node[^;]*?\{[^}]+\}", text))
    
    bad_patterns = check_bad_patterns(text)
    bad_pattern_count = len(bad_patterns["violations"])
    
    score = 0.0
    if has_tikz_environment: score += 0.25
    if valid_command_count >= 2: score += 0.25
    if coordinate_count >= 2: score += 0.15
    if non_preamble_command_ratio >= 0.70: score += 0.20
    if dom_ratio <= 0.40: score += 0.10
    if bad_pattern_count == 0: score += 0.05
    
    substantive_pass = score >= 0.75
    
    feats = SubstantiveFeatures(
        has_tikz_environment=has_tikz_environment,
        valid_command_count=valid_command_count,
        coordinate_count=coordinate_count,
        label_count=label_count,
        preamble_command_count=preamble_command_count,
        bad_pattern_count=bad_pattern_count,
        dominant_command_ratio=dom_ratio,
        non_preamble_command_ratio=non_preamble_command_ratio,
        substantive_score=score,
        substantive_pass=substantive_pass
    )
    return asdict(feats)


DEFAULT_GATE_CONFIG: dict[str, float] = {
    "promotion_min_compile_rate": 0.20,
    "hybrid_visual_score_threshold": 0.75,
    "hybrid_emd_threshold": 0.25,
    "render_diff_max_mean_abs": 0.03,
    "render_diff_max_changed_pixels": 0.05,
    "ablation_marker_hit_rate_min": 0.99,
    "production_marker_hit_rate_min": 0.90,
    "mask_zero_fraction_min": 0.01,
    "sentinel_repetition_loop_rate_max": 0.0,
    "repetition_loop_rate_max": 0.02,
    "truncation_rate_max": 0.10,
}

RECOVERY_CONTRACT_VERSION = PROMPT_CONTRACT_VERSION
RECOVERY_EVAL_SETS = ("sentinel_32", "ablation_100", "promotion_120", "stability_emd_32")
PREAMBLE_MARKER = "--- Starting Preamble ---"
DEFAULT_MODE_CAPS: dict[str, int | None] = {
    "plain_tikz": 4000,
    "pgfplots_axis": 4000,
    "scientific_schematic": 4000,
    "graph_nodes": None,
    "commutative_diagram": None,
}

COMMAND_RE = re.compile(r"\\[A-Za-z@]+")
HIGH_PRECISION_NUMBER_RE = re.compile(r"-?\d+\.\d{4,}")
ENV_RE = re.compile(r"\\begin\{(?:tikzpicture|tikz-cd|tikzcd|circuitikz|axis)\}", re.IGNORECASE)
STEALTH_STYLE_RE = re.compile(r">=\s*[sS]tealth,")


@dataclass(frozen=True, slots=True)
class QualityFilterConfig:
    max_token_length: int = 1536
    reject_metadata_truncated: bool = True
    stealth_loop_count: int = 10
    repeated_line_run: int = 5
    dominant_command_min_count: int = 20
    dominant_command_ratio: float = 0.40
    high_precision_line_ratio: float = 0.40


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    yield payload


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def record_mode(record: dict[str, Any]) -> str:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("generation_mode")
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _message_text(record: dict[str, Any], index: int) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list) or index >= len(messages):
        return ""
    message = messages[index]
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _role_texts(record: dict[str, Any], role: str) -> list[str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    return [
        _content_to_text(message.get("content"))
        for message in messages
        if isinstance(message, dict) and message.get("role") == role
    ]


def _single_role_text(record: dict[str, Any], role: str, fallback_index: int) -> str:
    values = _role_texts(record, role)
    if values:
        return "\n".join(values)
    return _message_text(record, fallback_index)


def _set_role_text(record: dict[str, Any], role: str, fallback_index: int, text: str) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return False
    target: dict[str, Any] | None = None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            target = message
            break
    if target is None and 0 <= fallback_index < len(messages) and isinstance(messages[fallback_index], dict):
        target = messages[fallback_index]
    if target is None:
        return False
    target["content"] = [{"type": "text", "text": text}]
    return True


def repair_assistant_contract(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    repaired = json.loads(json.dumps(record))
    original_user_text = _single_role_text(repaired, "user", 0)
    original_text = _single_role_text(repaired, "assistant", 1)

    description = original_user_text.strip()
    marker = "according to the following requirements:"
    marker_index = description.lower().find(marker)
    if marker_index >= 0:
        description = description[marker_index + len(marker):].strip()
    description = re.sub(r"```(?:latex)?", "", description, flags=re.IGNORECASE).strip()
    for stop_marker in (
        "\n\n[GEOMETRY HINTS]",
        "\n[GEOMETRY HINTS]",
        "\n\nOutput constraints:",
        "\nOutput constraints:",
        PREAMBLE_MARKER,
    ):
        stop_index = description.find(stop_marker)
        if stop_index >= 0:
            description = description[:stop_index].strip()

    metadata = repaired.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        repaired["metadata"] = metadata
    generation_mode = metadata.get("generation_mode")
    geometry_hints = metadata.get("geometry_hints")
    user_text = build_generation_prompt(
        description,
        generation_mode=generation_mode if isinstance(generation_mode, str) else None,
        geometry_hints=geometry_hints if isinstance(geometry_hints, dict) else None,
    )
    user_updated = _set_role_text(repaired, "user", 0, user_text)

    working = re.sub(r"```(?:latex)?", "", original_text, flags=re.IGNORECASE).strip()
    begin_match = re.search(r"\\begin\{document\}", working, flags=re.IGNORECASE)
    if begin_match:
        working = working[begin_match.end():]
    end_match = re.search(r"\\end\{document\}", working, flags=re.IGNORECASE)
    if end_match:
        working = working[: end_match.start()]

    env_match = re.search(r"\\begin\{(?:tikzpicture|tikz-cd|tikzcd|circuitikz|axis)\}", working, flags=re.IGNORECASE)
    if env_match:
        working = working[env_match.start():]

    assistant_text = working.strip() + "\n```\n"
    assistant_updated = _set_role_text(repaired, "assistant", 1, assistant_text)
    metadata["prompt_contract_version"] = PROMPT_CONTRACT_VERSION
    metadata["prompt_template_sha256"] = prompt_template_sha256()
    metadata["target_contract"] = "body_only_environment"

    audit = {
        "sample_id": str(record.get("sample_id", "")),
        "changed": original_user_text != user_text or original_text != assistant_text,
        "updated": user_updated and assistant_updated,
        "original_sha256": text_sha256(original_text),
        "repaired_sha256": text_sha256(assistant_text),
        "original_user_sha256": text_sha256(original_user_text),
        "repaired_user_sha256": text_sha256(user_text),
        "original_length": len(original_text),
        "repaired_length": len(assistant_text),
        "original_violations": validate_partial_decode_contract(record),
        "repaired_violations": validate_partial_decode_contract(repaired),
    }
    return repaired, audit


def validate_partial_decode_contract(record: dict[str, Any]) -> list[str]:
    user_texts = _role_texts(record, "user")
    assistant_texts = _role_texts(record, "assistant")
    user_text = "\n".join(user_texts) if user_texts else _message_text(record, 0)
    assistant_text = "\n".join(assistant_texts) if assistant_texts else _message_text(record, 1)
    violations: list[str] = []
    if len(user_texts) != 1:
        violations.append("user_count_not_one")
    if len(assistant_texts) != 1:
        violations.append("assistant_count_not_one")
    if user_text.count("```latex") != 1:
        violations.append("opening_latex_fence_count")
    user_marker_violations = {
        r"\\documentclass(?:\[[^\]]*\])?\{": "user_documentclass",
        r"\\begin\{document\}": "user_begin_document",
        r"\\end\{document\}": "user_end_document",
        r"\\usepackage(?:\[[^\]]*\])?\{": "user_usepackage",
        r"\\PreviewEnvironment\{": "user_preview_environment",
        r"\\usetikzlibrary\{": "user_usetikzlibrary",
        re.escape(PREAMBLE_MARKER): "user_preamble_marker",
    }
    for pattern, violation in user_marker_violations.items():
        if re.search(pattern, user_text):
            violations.append(violation)
    if assistant_text.count("```") != 1:
        violations.append("closing_fence_count")
    if not re.match(
        r"^\s*\\begin\{(?:tikzpicture|tikz-cd|tikzcd|circuitikz|axis|tikzpicture\*)\}",
        assistant_text,
        flags=re.IGNORECASE,
    ):
        violations.append("assistant_env_start")
    marker_violations = {
        "\\documentclass": "assistant_documentclass",
        "\\begin{document}": "assistant_begin_document",
        "\\end{document}": "assistant_end_document",
        "\\usepackage": "assistant_usepackage",
        "\\PreviewEnvironment": "assistant_preview_environment",
        "\\usetikzlibrary": "assistant_usetikzlibrary",
        PREAMBLE_MARKER: "assistant_preamble_marker",
        "```latex": "assistant_opening_latex_fence",
    }
    for marker, violation in marker_violations.items():
        if marker in assistant_text:
            violations.append(violation)
    return violations


def validate_contract_file(path: Path, *, limit: int | None = None) -> dict[str, Any]:
    checked = 0
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(iter_jsonl(path)):
        if limit is not None and checked >= limit:
            break
        checked += 1
        violations = validate_partial_decode_contract(record)
        if violations:
            failures.append(
                {
                    "row_index": index,
                    "sample_id": _sample_id(record, index),
                    "violations": violations,
                }
            )
    return {
        "checked": checked,
        "failure_count": len(failures),
        "failures": failures[:50],
    }


def repetition_features(text: str, config: QualityFilterConfig | None = None) -> dict[str, Any]:
    config = config or QualityFilterConfig()
    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]
    max_line_run = 0
    current_run = 0
    previous_line: str | None = None
    repeated_line = ""
    for line in stripped_lines:
        if line == previous_line:
            current_run += 1
        else:
            current_run = 1
            previous_line = line
        if current_run > max_line_run:
            max_line_run = current_run
            repeated_line = line

    commands = COMMAND_RE.findall(text)
    dominant_command = None
    dominant_command_count = 0
    dominant_command_ratio = 0.0
    if commands:
        dominant_command, dominant_command_count = Counter(commands).most_common(1)[0]
        dominant_command_ratio = dominant_command_count / len(commands)

    stealth_count = len(STEALTH_STYLE_RE.findall(text))
    has_dominant_command_loop = (
        len(commands) >= config.dominant_command_min_count
        and dominant_command_ratio > config.dominant_command_ratio
    )
    has_repetition_loop = (
        stealth_count >= config.stealth_loop_count
        or max_line_run >= config.repeated_line_run
    )
    return {
        "stealth_loop_count": stealth_count,
        "max_repeated_line_run": max_line_run,
        "repeated_line": repeated_line,
        "command_count": len(commands),
        "dominant_command": dominant_command,
        "dominant_command_count": dominant_command_count,
        "dominant_command_ratio": dominant_command_ratio,
        "has_dominant_command_loop": has_dominant_command_loop,
        "has_repetition_loop": has_repetition_loop,
    }


def has_repetition_failure(text: str, config: QualityFilterConfig | None = None) -> bool:
    return bool(repetition_features(text, config)["has_repetition_loop"])


def _structure_violations(text: str) -> list[str]:
    violations: list[str] = []
    for label, opener, closer in (
        ("unbalanced_braces", "{", "}"),
        ("unbalanced_brackets", "[", "]"),
        ("unbalanced_parentheses", "(", ")"),
    ):
        balance = 0
        min_balance = 0
        for char in text:
            if char == opener:
                balance += 1
            elif char == closer:
                balance -= 1
                min_balance = min(min_balance, balance)
        if balance != 0 or min_balance < 0:
            violations.append(label)
    return violations


def quality_filter_record(
    record: dict[str, Any],
    *,
    config: QualityFilterConfig | None = None,
) -> tuple[list[str], dict[str, Any]]:
    config = config or QualityFilterConfig()
    reasons: list[str] = []
    features: dict[str, Any] = {}

    contract_violations = validate_partial_decode_contract(record)
    reasons.extend(f"contract:{item}" for item in contract_violations)

    assistant_text = _single_role_text(record, "assistant", 1)
    if not ENV_RE.search(assistant_text):
        reasons.append("missing_tikz_environment")

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        token_length = metadata.get("token_length")
        if isinstance(token_length, int):
            features["token_length"] = token_length
            if token_length > config.max_token_length:
                reasons.append("token_length_exceeds_max")
        if config.reject_metadata_truncated and metadata.get("is_truncated") is True:
            reasons.append("metadata_truncated")

    rep = repetition_features(assistant_text, config)
    features["repetition"] = rep
    if rep["stealth_loop_count"] >= config.stealth_loop_count:
        reasons.append("stealth_loop")
    if rep["max_repeated_line_run"] >= config.repeated_line_run:
        reasons.append("repeated_line_loop")
    # We no longer reject for dominant_command_loop as it is too aggressive for TikZ

    nonempty_lines = [line for line in assistant_text.splitlines() if line.strip()]
    high_precision_lines = [line for line in nonempty_lines if HIGH_PRECISION_NUMBER_RE.search(line)]
    high_precision_ratio = len(high_precision_lines) / len(nonempty_lines) if nonempty_lines else 0.0
    features["high_precision_numeric_line_ratio"] = high_precision_ratio
    features["high_precision_numeric_lines"] = len(high_precision_lines)
    if high_precision_ratio > config.high_precision_line_ratio:
        reasons.append("high_precision_numeric_poisoning")

    structure_violations = _structure_violations(assistant_text)
    reasons.extend(structure_violations)
    features["structure_violations"] = structure_violations
    return reasons, features


def filter_quality_records(
    records: Iterable[dict[str, Any]],
    *,
    config: QualityFilterConfig | None = None,
    max_rejection_examples: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = config or QualityFilterConfig()
    kept: list[dict[str, Any]] = []
    total = 0
    reason_counts: Counter[str] = Counter()
    input_modes: Counter[str] = Counter()
    output_modes: Counter[str] = Counter()
    rejection_examples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        total += 1
        mode = record_mode(record)
        input_modes[mode] += 1
        reasons, features = quality_filter_record(record, config=config)
        if reasons:
            reason_counts.update(reasons)
            if len(rejection_examples) < max_rejection_examples:
                rejection_examples.append(
                    {
                        "row_index": index,
                        "sample_id": _sample_id(record, index),
                        "mode": mode,
                        "reasons": reasons,
                        "features": features,
                    }
                )
            continue
        kept.append(record)
        output_modes[mode] += 1
    audit = {
        "total_input": total,
        "total_output": len(kept),
        "total_rejected": total - len(kept),
        "quality_filter_config": asdict(config),
        "reason_counts": dict(sorted(reason_counts.items())),
        "input_mode_counts": dict(sorted(input_modes.items())),
        "output_mode_counts": dict(sorted(output_modes.items())),
        "rejection_examples": rejection_examples,
    }
    audit["quality_filter_config_hash"] = stable_json_sha256(audit["quality_filter_config"])
    return kept, audit


def select_mode_capped_records(
    records: list[dict[str, Any]],
    *,
    caps: dict[str, int | None] | None = None,
    seed: int = 20260427,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    caps = DEFAULT_MODE_CAPS if caps is None else caps
    by_mode: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        by_mode[record_mode(record)].append((index, record))

    rng = random.Random(seed)
    selected_indexes: set[int] = set()
    mode_counts_before: dict[str, int] = {}
    mode_counts_after: dict[str, int] = {}
    for mode, values in by_mode.items():
        mode_counts_before[mode] = len(values)
        shuffled = sorted(values, key=lambda item: _sample_id(item[1], item[0]))
        rng.shuffle(shuffled)
        cap = caps.get(mode)
        chosen = shuffled if cap is None else shuffled[: max(0, cap)]
        mode_counts_after[mode] = len(chosen)
        selected_indexes.update(index for index, _ in chosen)

    selected = [record for index, record in enumerate(records) if index in selected_indexes]
    audit = {
        "seed": seed,
        "mode_caps": caps,
        "total_input": len(records),
        "total_output": len(selected),
        "input_mode_counts": dict(sorted(mode_counts_before.items())),
        "output_mode_counts": dict(sorted(mode_counts_after.items())),
    }
    audit["mode_balance_config_hash"] = stable_json_sha256({"seed": seed, "mode_caps": caps})
    return selected, audit


def synthetic_repetition_examples() -> list[dict[str, Any]]:
    snippets = [
        "\n".join(["    >=stealth,"] * 12),
        "\n".join([r"\draw (0,0) -- (1,1);"] * 6),
        "\n".join([r"\node at (0,0) {x};"] * 24),
    ]
    examples: list[dict[str, Any]] = []
    for index, text in enumerate(snippets):
        examples.append(
            {
                "sample_id": f"synthetic_repetition_{index:03d}",
                "source": "synthetic",
                "text": text,
                "features": repetition_features(text),
            }
        )
    return examples


def _sample_id(record: dict[str, Any], index: int) -> str:
    value = record.get("sample_id")
    return str(value) if value not in (None, "") else f"row_{index:06d}"


def _quota_by_mode(modes: list[str], total: int) -> dict[str, int]:
    if not modes:
        return {}
    base = total // len(modes)
    remainder = total % len(modes)
    return {mode: base + (1 if idx < remainder else 0) for idx, mode in enumerate(sorted(modes))}


def select_equal_mode_sample_ids(
    records: list[dict[str, Any]],
    *,
    total: int,
    seed: int,
    excluded: set[str] | None = None,
    min_remaining_per_mode: int = 0,
) -> list[str]:
    if min_remaining_per_mode < 0:
        raise ValueError("min_remaining_per_mode must be non-negative.")
    excluded = set() if excluded is None else set(excluded)
    by_mode: dict[str, list[str]] = defaultdict(list)
    for index, record in enumerate(records):
        sample_id = _sample_id(record, index)
        if sample_id in excluded:
            continue
        by_mode[record_mode(record)].append(sample_id)

    rng = random.Random(seed)
    for values in by_mode.values():
        values.sort()
        rng.shuffle(values)

    quotas = _quota_by_mode(list(by_mode), total)
    selected: list[str] = []
    remaining: dict[str, list[str]] = {}
    deficit = 0
    for mode, quota in quotas.items():
        values = by_mode.get(mode, [])
        selectable_count = max(0, len(values) - min_remaining_per_mode)
        take = min(quota, selectable_count)
        selected.extend(values[:take])
        remaining[mode] = values[take:selectable_count]
        deficit += quota - take

    while deficit > 0:
        candidates = [
            mode for mode, values in sorted(remaining.items(), key=lambda item: (len(item[1]), item[0]))
            if values
        ]
        if not candidates:
            break
        for mode in candidates:
            if deficit <= 0:
                break
            selected.append(remaining[mode].pop(0))
            deficit -= 1

    return selected


def build_eval_manifest(
    dataset_path: Path,
    *,
    seed: int,
    prompt_contract_version: str = RECOVERY_CONTRACT_VERSION,
    decoding_config: dict[str, Any] | None = None,
    compiler_config: dict[str, Any] | None = None,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    records = list(iter_jsonl(dataset_path))
    used: set[str] = set()
    set_specs = {
        "sentinel_32": 32,
        "ablation_100": 100,
        "promotion_120": 120,
        "stability_emd_32": 32,
    }
    sets: dict[str, list[str]] = {}
    for offset, (name, size) in enumerate(set_specs.items()):
        sample_ids = select_equal_mode_sample_ids(records, total=size, seed=seed + offset, excluded=used)
        sets[name] = sample_ids
        used.update(sample_ids)

    mode_counts: dict[str, dict[str, int]] = {}
    by_id = {_sample_id(record, index): record for index, record in enumerate(records)}
    for name, sample_ids in sets.items():
        counts: dict[str, int] = defaultdict(int)
        for sample_id in sample_ids:
            counts[record_mode(by_id[sample_id])] += 1
        mode_counts[name] = dict(sorted(counts.items()))

    payload = {
        "schema_version": 1,
        "dataset_path": str(dataset_path.expanduser().resolve()),
        "dataset_sha256": file_sha256(dataset_path),
        "seed": seed,
        "prompt_contract_version": prompt_contract_version,
        "decoding_config": decoding_config or {},
        "compiler_config": compiler_config or {},
        "max_tokens": max_tokens,
        "sets": sets,
        "mode_counts": mode_counts,
    }
    payload["manifest_sha256"] = stable_json_sha256({k: v for k, v in payload.items() if k != "manifest_sha256"})
    return payload


def assert_manifest_sets_disjoint(manifest: dict[str, Any]) -> None:
    sets = manifest.get("sets")
    if not isinstance(sets, dict):
        raise RuntimeError("Eval manifest missing sets.")
    owner: dict[str, str] = {}
    overlaps: list[tuple[str, str, str]] = []
    for name in RECOVERY_EVAL_SETS:
        values = sets.get(name)
        if not isinstance(values, list):
            raise RuntimeError(f"Eval manifest missing set {name}.")
        for value in values:
            sample_id = str(value)
            previous = owner.get(sample_id)
            if previous is not None:
                overlaps.append((sample_id, previous, name))
            owner[sample_id] = name
    if overlaps:
        raise RuntimeError(f"Eval manifest sets are not disjoint: {overlaps[:5]}")


def validate_split_disjoint(paths: dict[str, Path]) -> dict[str, int]:
    owners: dict[str, str] = {}
    overlaps: list[tuple[str, str, str]] = []
    counts: dict[str, int] = {}
    for split, path in paths.items():
        seen = 0
        for index, record in enumerate(iter_jsonl(path)):
            sample_id = _sample_id(record, index)
            seen += 1
            previous = owners.get(sample_id)
            if previous is not None:
                overlaps.append((sample_id, previous, split))
            owners[sample_id] = split
        counts[split] = seen
    if overlaps:
        raise RuntimeError(f"Split sample_id overlap detected: {overlaps[:10]}")
    return counts


def write_gate_config(path: Path, *, overwrite: bool = False) -> dict[str, float]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Gate config already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_GATE_CONFIG, indent=2, sort_keys=True), encoding="utf-8")
    return DEFAULT_GATE_CONFIG


def evaluate_ab_result_gate(
    results_payload: dict[str, Any],
    *,
    candidate_key: str,
    gate: str,
    base_key: str = "base",
    gate_config: dict[str, float] | None = None,
) -> dict[str, Any]:
    gate_config = DEFAULT_GATE_CONFIG if gate_config is None else gate_config
    candidate = results_payload.get(candidate_key)
    if not isinstance(candidate, dict):
        raise RuntimeError(f"Missing candidate metrics block: {candidate_key}")
    base = results_payload.get(base_key)
    if not isinstance(base, dict):
        base = {}

    checks: dict[str, dict[str, Any]] = {}
    repetition_key = (
        "sentinel_repetition_loop_rate_max"
        if gate == "sentinel"
        else "repetition_loop_rate_max"
    )
    repetition_ceiling = gate_config.get(repetition_key, 0.0 if gate == "sentinel" else 0.02)
    candidate_repetition = candidate.get("repetition_loop_rate")
    checks["repetition_loop_rate"] = {
        "required": repetition_ceiling,
        "observed": candidate_repetition,
        "passed": isinstance(candidate_repetition, (int, float)) and candidate_repetition <= repetition_ceiling,
    }

    truncation_ceiling = gate_config.get("truncation_rate_max", 0.10)
    candidate_truncation = candidate.get("truncation_rate")
    checks["truncation_rate"] = {
        "required": truncation_ceiling,
        "observed": candidate_truncation,
        "passed": isinstance(candidate_truncation, (int, float)) and candidate_truncation <= truncation_ceiling,
    }

    if gate in {"ablation", "promotion"}:
        compile_floor = gate_config.get("promotion_min_compile_rate", 0.20)
        base_compile = base.get("compile_rate")
        if isinstance(base_compile, (int, float)):
            compile_floor = max(compile_floor, 0.5 * float(base_compile))
        candidate_compile = candidate.get("compile_rate")
        checks["compile_rate"] = {
            "required": compile_floor,
            "observed": candidate_compile,
            "passed": isinstance(candidate_compile, (int, float)) and candidate_compile >= compile_floor,
        }

    return {
        "gate": gate,
        "candidate_key": candidate_key,
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


@dataclass(slots=True)
class StabilityCheckpoint:
    checkpoint_path: str
    iteration: int
    collapse: bool
    partial_emd: float
    compile_rate: float
    failure_rate: float


def select_stability_checkpoint(
    candidates: list[StabilityCheckpoint],
    *,
    base_stability_emd: float,
) -> StabilityCheckpoint:
    if not any(item.iteration == 5000 for item in candidates):
        raise RuntimeError("The 5000-iteration checkpoint must be evaluated before stability selection.")
    eligible = [
        item for item in candidates
        if not item.collapse and item.partial_emd <= base_stability_emd
    ]
    if not eligible:
        raise RuntimeError("No stability checkpoint is eligible for promotion.")
    eligible.sort(
        key=lambda item: (
            item.partial_emd,
            -item.compile_rate,
            item.failure_rate,
            -item.iteration,
        )
    )
    return eligible[0]


def stability_checkpoint_from_dict(payload: dict[str, Any]) -> StabilityCheckpoint:
    return StabilityCheckpoint(
        checkpoint_path=str(payload["checkpoint_path"]),
        iteration=int(payload["iteration"]),
        collapse=bool(payload.get("collapse", False)),
        partial_emd=float(payload["partial_emd"]),
        compile_rate=float(payload["compile_rate"]),
        failure_rate=float(payload.get("failure_rate", 0.0)),
    )


def validate_sweep_resume_adapter(adapter_metadata_path: Path | None, archive_manifest_path: Path | None) -> None:
    if adapter_metadata_path is None:
        return
    if archive_manifest_path is None or not archive_manifest_path.exists():
        raise RuntimeError("No resume_adapter_path is allowed when the promotion archive manifest is absent.")
    metadata = json.loads(adapter_metadata_path.read_text(encoding="utf-8"))
    if metadata.get("promoted") is not True:
        raise RuntimeError("Sweep resume adapter metadata must contain promoted=true.")
    adapter_sha256 = metadata.get("adapter_sha256")
    archive = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
    entries = archive.get("entries", [])
    if not isinstance(entries, list) or not any(entry.get("sha256") == adapter_sha256 for entry in entries if isinstance(entry, dict)):
        raise RuntimeError("Sweep resume adapter SHA256 is not present in the promotion archive manifest.")


def as_jsonable_checkpoint(checkpoint: StabilityCheckpoint) -> dict[str, Any]:
    return asdict(checkpoint)
