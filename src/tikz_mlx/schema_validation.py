from __future__ import annotations

import re

DOCUMENT_CLASS_TIKZ_RE = re.compile(r"\\documentclass\s*\[tikz\]\s*\{standalone\}", re.IGNORECASE)
ANY_DOCUMENT_CLASS_RE = re.compile(r"\\documentclass\b", re.IGNORECASE)
BEGIN_DOC_RE = re.compile(r"\\begin\{document\}", re.IGNORECASE)
END_DOC_RE = re.compile(r"\\end\{document\}", re.IGNORECASE)


def is_canonical_tikz_document(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    # Hard reject markdown-fenced payloads.
    if "```" in stripped:
        return False

    class_matches = list(DOCUMENT_CLASS_TIKZ_RE.finditer(stripped))
    any_class_matches = list(ANY_DOCUMENT_CLASS_RE.finditer(stripped))
    begin_matches = list(BEGIN_DOC_RE.finditer(stripped))
    end_matches = list(END_DOC_RE.finditer(stripped))

    # Hard reject duplicated wrappers or non-canonical documentclass lines.
    if len(class_matches) != 1 or len(any_class_matches) != 1:
        return False
    if len(begin_matches) != 1 or len(end_matches) != 1:
        return False

    class_match = class_matches[0]
    begin_match = begin_matches[0]
    end_match = end_matches[0]

    if class_match.start() > begin_match.start():
        return False
    if begin_match.start() > end_match.start():
        return False

    leading = stripped[: class_match.start()]
    trailing = stripped[end_match.end() :]
    if leading.strip() or trailing.strip():
        return False

    return True
