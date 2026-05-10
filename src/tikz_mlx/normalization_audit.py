import difflib
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any
from tikz_mlx.normalize import normalize_tikz

@dataclass
class NormalizationAudit:
    raw_length: int
    normalized_length: int
    edit_distance_ratio: float
    inserted_semicolon_count: int
    closed_environment_count: int
    removed_duplicate_line_count: int
    added_package_count: int
    suspicious: bool
    reasons: List[str]

def normalize_with_audit(raw: str) -> Tuple[str, Dict[str, Any]]:
    normalized = normalize_tikz(raw)
    
    raw_len = len(raw)
    norm_len = len(normalized)
    
    # difflib.SequenceMatcher.ratio() returns a float in [0, 1]
    # where 1.0 means identical. So distance ratio is 1 - ratio.
    ratio = difflib.SequenceMatcher(None, raw, normalized).ratio()
    edit_distance_ratio = 1.0 - ratio
    
    inserted_semicolon_count = normalized.count(";") - raw.count(";")
    closed_env_count = normalized.count("\\end{") - raw.count("\\end{")
    added_pkg_count = normalized.count("\\usepackage") - raw.count("\\usepackage")
    
    # heuristic for duplicate lines removed
    raw_lines = len(raw.splitlines())
    norm_lines = len(normalized.splitlines())
    removed_duplicate_line_count = max(0, raw_lines - norm_lines)
    
    suspicious = False
    reasons = []
    
    if edit_distance_ratio > 0.40:
        suspicious = True
        reasons.append(f"High edit distance ratio: {edit_distance_ratio:.2f}")
    if raw_len > 0 and norm_len / raw_len > 2.0:
        suspicious = True
        reasons.append("Normalized length is more than double the raw length")
    if removed_duplicate_line_count > 10:
        suspicious = True
        reasons.append(f"Removed {removed_duplicate_line_count} lines")
    if added_pkg_count > 5:
        suspicious = True
        reasons.append(f"Added {added_pkg_count} packages")
    if closed_env_count > 3:
        suspicious = True
        reasons.append(f"Closed {closed_env_count} environments")
        
    audit = NormalizationAudit(
        raw_length=raw_len,
        normalized_length=norm_len,
        edit_distance_ratio=edit_distance_ratio,
        inserted_semicolon_count=max(0, inserted_semicolon_count),
        closed_environment_count=max(0, closed_env_count),
        removed_duplicate_line_count=removed_duplicate_line_count,
        added_package_count=max(0, added_pkg_count),
        suspicious=suspicious,
        reasons=reasons
    )
    
    return normalized, asdict(audit)
