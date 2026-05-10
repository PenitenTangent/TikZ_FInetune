from dataclasses import dataclass
import re
from typing import Dict, Any, List

@dataclass
class Rule:
    name: str
    pattern: str
    max_count: int

BAD_PATTERN_RULES = [
    Rule("repeated_preview_environment", pattern=r"\\PreviewEnvironment", max_count=0),
    Rule("zero_geometric", pattern=r"0\.geometric", max_count=0),
    Rule("decorations_geometric_loop", pattern=r"decorations\.geometric", max_count=5),
    Rule("assistant_usepackage", pattern=r"\\usepackage", max_count=0),
    Rule("assistant_documentclass", pattern=r"\\documentclass", max_count=0),
    Rule("opening_latex_fence", pattern=r"```latex", max_count=0),
    Rule("repeated_draw_node_excessive", pattern=r"\\(?:draw|node|path)", max_count=100),
    Rule("consecutive_backslashes", pattern=r"\\{4,}", max_count=0),
]

def check_bad_patterns(text: str) -> Dict[str, Any]:
    """Scans text for known bad pattern regressions."""
    violations = []
    counts = {}
    
    for rule in BAD_PATTERN_RULES:
        count = len(re.findall(rule.pattern, text))
        counts[rule.name] = count
        if count > rule.max_count:
            violations.append({
                "rule": rule.name,
                "count": count,
                "max_allowed": rule.max_count
            })
            
    return {
        "pass": len(violations) == 0,
        "violations": violations,
        "counts": counts
    }
