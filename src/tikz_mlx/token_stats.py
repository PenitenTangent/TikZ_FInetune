import re
from collections import Counter
from typing import Dict, Any, Tuple, Optional

COMMAND_RE = re.compile(r"\\[A-Za-z@]+")
COORD_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")

BAD_PATTERNS = {
    "preview_environment": r"\\PreviewEnvironment",
    "usepackage": r"\\usepackage",
    "documentclass": r"\\documentclass",
    "begin_document": r"\\begin\{document\}",
    "end_document": r"\\end\{document\}",
    "zero_geometric": r"0\.geometric",
    "decorations_geometric": r"decorations\.geometric",
}

def command_counts(text: str) -> Counter:
    return Counter(COMMAND_RE.findall(text))

def bad_pattern_counts(text: str) -> Dict[str, int]:
    return {
        name: len(re.findall(pattern, text))
        for name, pattern in BAD_PATTERNS.items()
    }

def dominant_command_ratio(text: str) -> Tuple[Optional[str], int, float]:
    counts = command_counts(text)
    if not counts:
        return None, 0, 0.0
    total = sum(counts.values())
    dominant, max_count = counts.most_common(1)[0]
    return dominant, max_count, max_count / total

def boilerplate_score(text: str) -> float:
    # Weighted score from bad patterns + preamble commands + dominant command ratio
    score = 0.0
    patterns = bad_pattern_counts(text)
    for name, count in patterns.items():
        if name in ["preview_environment", "zero_geometric", "decorations_geometric"]:
            score += count * 0.5
        elif name in ["usepackage", "documentclass", "begin_document", "end_document"]:
            score += count * 0.2
            
    _, _, ratio = dominant_command_ratio(text)
    if ratio > 0.4:
        score += (ratio - 0.4) * 2.0
        
    return score

def token_distribution_features(text: str) -> Dict[str, Any]:
    dom_cmd, dom_count, dom_ratio = dominant_command_ratio(text)
    return {
        "command_counts": dict(command_counts(text)),
        "bad_pattern_counts": bad_pattern_counts(text),
        "dominant_command": dom_cmd,
        "dominant_command_count": dom_count,
        "dominant_command_ratio": dom_ratio,
        "boilerplate_score": boilerplate_score(text),
        "coordinate_count": len(COORD_RE.findall(text)),
    }
