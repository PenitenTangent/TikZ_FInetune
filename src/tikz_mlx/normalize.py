from __future__ import annotations

import re


DOCUMENT_CLASS = r"\documentclass{article}"
DOCUMENT_BEGIN = r"\begin{document}"
DOCUMENT_END = r"\end{document}"

EXTERNAL_DEPENDENCY_PATTERNS = (
    r"\\input\{[^}]+\}",
    r"\\include\{[^}]+\}",
    r"\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}",
)

PACKAGE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\\begin\{tikz-cd\}|\\arrow\b"), r"\usepackage{tikz-cd}"),
    (re.compile(r"\\begin\{circuitikz\}|to\[[^]]*resistor"), r"\usepackage{circuitikz}"),
    (re.compile(r"\\begin\{axis\}|\\begin\{semilog[xy]axis\}|\\addplot\b|\\pgfplotsset\b|\\pgfplotstable"), r"\usepackage{pgfplots}"),
)


def strip_inline_comments(text: str) -> str:
    """Remove TeX inline comments (starting with %) that are not escaped.

    Also rstrips each line to remove trailing whitespace.
    """
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"(?<!\\)%.*$", "", line)
        lines.append(cleaned.rstrip())
    return "\n".join(lines).strip()


def strip_external_dependency_lines(text: str) -> str:
    result = text
    for pattern in EXTERNAL_DEPENDENCY_PATTERNS:
        result = re.sub(rf"^.*{pattern}.*$\n?", "", result, flags=re.MULTILINE)
    return result.strip()


def contains_external_dependencies(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in EXTERNAL_DEPENDENCY_PATTERNS)


def detect_required_packages(text: str) -> list[str]:
    packages = {
        r"\usepackage[active,tightpage]{preview}",
        r"\usepackage{tikz}",
        r"\PreviewEnvironment{tikzpicture}",
        r"\PreviewEnvironment{tikz-cd}",
        r"\PreviewEnvironment{circuitikz}",
        r"\PreviewEnvironment{axis}",
    }
    # Add most common TikZ libraries to ensure compilation of modern syntax
    packages.add(r"\usetikzlibrary{positioning, arrows.meta, calc, shapes.geometric, intersections, decorations.pathreplacing, decorations.markings, backgrounds, quotes, fit, matrix, patterns, through, hobby}")

    for pattern, package in PACKAGE_RULES:
        if re.search(pattern, text):
            packages.add(package)

    if r"\usepackage{pgfplots}" in packages:
        packages.add(r"\pgfplotsset{compat=1.18}")
    
    # Special fix for pgfplots which often needs a compat level
    # Ensure preview is loaded before its settings
    ordered = []
    if r"\usepackage[active,tightpage]{preview}" in packages:
        ordered.append(r"\usepackage[active,tightpage]{preview}")
    
    # Add other usepackages
    for p in sorted(packages):
        if p.startswith(r"\usepackage") and p not in ordered:
            ordered.append(p)
            
    # Add settings and libraries
    for p in sorted(packages):
        if not p.startswith(r"\usepackage"):
            ordered.append(p)
            
    return ordered


def heal_coordinate_math(text: str) -> str:
    """Ensure coordinate math is wrapped in braces.

    Fixes cases like (x+1, y) -> ({x+1}, {y}) which TikZ requires for 
    complex expressions inside coordinates.
    """
    result = []
    i = 0
    while i < len(text):
        if text[i:i+2] == '($':
            result.append(text[i:i+2])
            i += 2
            continue
        if text[i] == '(':
            start = i
            depth = 1
            j = i + 1
            while j < len(text) and depth > 0:
                if text[j] == '(': depth += 1
                elif text[j] == ')': depth -= 1
                j += 1
            if depth == 0:
                inner = text[start+1:j-1]
                if ',' in inner and not ':' in inner:
                    parts = []
                    current_part = []
                    inner_depth = 0
                    for char in inner:
                        if char == '(': inner_depth += 1
                        elif char == ')': inner_depth -= 1
                        if char == ',' and inner_depth == 0:
                            parts.append("".join(current_part))
                            current_part = []
                        else:
                            current_part.append(char)
                    parts.append("".join(current_part))
                    math_chars = set("+-*/\\")
                    new_parts = []
                    for p in parts:
                        ps = p.strip()
                        if ps and not (ps.startswith('{') and ps.endswith('}')):
                            if any(c in math_chars for c in ps) or any(func in ps for func in ['sin', 'cos', 'tan']):
                                p = p.replace(ps, f"{{{ps}}}")
                        new_parts.append(p)
                    new_inner = ",".join(new_parts)
                    result.append(f"({new_inner})")
                else:
                    result.append(text[start:j])
                i = j
                continue
        result.append(text[i])
        i += 1
    return "".join(result)


def heal_missing_semicolons(text: str) -> str:
    r"""Insert missing semicolons at the end of TikZ commands.

    Heals cases where the model forgets the mandatory semicolon at the end
    of \draw, \node, etc., by detecting the start of the next command.
    """
    commands = [r"\\draw", r"\\fill", r"\\path", r"\\node", r"\\coordinate", 
                r"\\filldraw", r"\\clip", r"\\begin\{[^}]+\}", r"\\end\{[^}]+\}"]
    pattern = re.compile(r'(?<!\\)(' + '|'.join(commands) + r')(?![a-zA-Z])')
    parts = pattern.split(text)
    if len(parts) == 1:
        return text
    result = [parts[0]]
    for i in range(1, len(parts), 2):
        cmd = parts[i]
        after = parts[i+1]
        prev_text = result[-1]
        j = len(prev_text) - 1
        while j >= 0 and prev_text[j].isspace():
            j -= 1
        if j >= 0:
            last_char = prev_text[j]
            if last_char not in (';', '{', '[', '%', '>'):
                if last_char == '}':
                    begin_match = re.search(r'\\begin\{[^}]+\}\s*$', prev_text[:j+1])
                    if not begin_match:
                        result[-1] = prev_text[:j+1] + ';' + prev_text[j+1:]
                else:
                    result[-1] = prev_text[:j+1] + ';' + prev_text[j+1:]
        result.append(cmd)
        result.append(after)
    return "".join(result)


def heal_broken_pgfplots(text: str) -> str:
    """Fix common pgfplots hallucinations from language models.

    Specifically handles the non-existent 'pgfplotstabletypeset' environment 
    and invalid layer declaration forms.
    """
    # \begin{pgfplotstabletypeset} is not a real environment;
    # rewrite to the correct command form: \pgfplotstabletypeset{
    text = re.sub(
        r"\\begin\{pgfplotstabletypeset\}",
        r"\\pgfplotstabletypeset{",
        text,
    )
    text = re.sub(
        r"\\end\{pgfplotstabletypeset\}",
        r"}",
        text,
    )
    # Drop invalid \pgfdeclarelayer{.../.cd}{} forms
    text = re.sub(
        r"\\pgfdeclarelayer\{[^}]*/\.cd\}\{\}",
        "",
        text,
    )
    return text


def heal_unclosed_environments(text: str) -> str:
    """Auto-close unmatched TikZ/pgfplots environments."""
    envs = ["tikzpicture", "axis", "semilogxaxis", "semilogyaxis",
            "circuitikz", "tikz-cd", "scope"]
    for env in envs:
        opens = len(re.findall(rf"\\begin\{{{re.escape(env)}\}}", text))
        closes = len(re.findall(rf"\\end\{{{re.escape(env)}\}}", text))
        while closes < opens:
            text += f"\n\\end{{{env}}}"
            closes += 1
    return text


def unwrap_document(text: str) -> str:
    stripped = text.strip()
    # Find all \begin{document} and \end{document}
    begins = list(re.finditer(r"\\begin\{document\}", stripped, flags=re.IGNORECASE))
    ends = list(re.finditer(r"\\end\{document\}", stripped, flags=re.IGNORECASE))
    
    if begins and ends:
        # We take the content between the LAST \begin{document} and the FIRST \end{document}
        # to heal double-preamble artifacts.
        start_idx = begins[-1].end()
        end_idx = ends[0].start()
        if start_idx < end_idx:
            return stripped[start_idx:end_idx].strip()
            
    # Fallback: if no document tags, try to strip until first TikZ environment
    env_match = re.search(r"\\begin\{(?:tikzpicture|tikz-cd|circuitikz|axis)\}", stripped, flags=re.IGNORECASE)
    if env_match:
        content = stripped[env_match.start():]
        # Trim any trailing \end{document}
        content = re.sub(r"\\end\{document\}.*", "", content, flags=re.IGNORECASE | re.DOTALL)
        return content.strip()
        
    return stripped


def ensure_standalone_document(text: str) -> str:
    # Unwrap existing document if present
    body = unwrap_document(strip_inline_comments(text))
    # Eliminate double newlines that cause paragraph break errors in TikZ parsers
    body = re.sub(r"\n\s*\n", "\n", body)
    
    # Heal common syntax hallucinations from language models
    body = heal_broken_pgfplots(body)
    body = heal_coordinate_math(body)
    body = heal_missing_semicolons(body)
    body = heal_unclosed_environments(body)
    
    # Modernize tikzstyle to tikzset for robustness
    # Handles both \tikzstyle{name} = [style] and \tikzstyle{name} = {style}
    body = re.sub(r"\\tikzstyle\{([^}]+)\}\s*=\s*[\[{]([^\]}]+)[\]}]", r"\\tikzset{\1/.style={\2}}", body)

    packages = "\n".join(detect_required_packages(body))
    return f"{DOCUMENT_CLASS}\n{packages}\n{DOCUMENT_BEGIN}\n{body}\n{DOCUMENT_END}"


# TikZ renderer defaults — these options are always active even when omitted.
# Stripping them saves ~2–5 tokens/command and removes learned verbosity.
_DEFAULT_OPTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r",?\s*draw\s*=\s*black(?=\s*[,\]])\b"),
    re.compile(r",?\s*(?:color|text)\s*=\s*black(?=\s*[,\]])\b"),
    re.compile(r",?\s*fill\s*=\s*white(?=\s*[,\]])\b"),  # white fill is default
    re.compile(r",?\s*line\s+width\s*=\s*0\.4\s*pt(?=\s*[,\]])\b"),
    re.compile(r",?\s*line\s+cap\s*=\s*butt(?=\s*[,\]])\b"),
    re.compile(r",?\s*line\s+join\s*=\s*miter(?=\s*[,\]])\b"),
    re.compile(r",?\s*miter\s+limit\s*=\s*10(?=\s*[,\]])\b"),
    re.compile(r",?\s*dash\s+phase\s*=\s*0(?:pt)?(?=\s*[,\]])\b"),
    re.compile(r",?\s*scale\s*=\s*1(?:\.0)?(?=\s*[,\]])\b"),
    re.compile(r",?\s*(?:text |fill |draw )?opacity\s*=\s*1(?:\.0)?(?=\s*[,\]])\b"),
    re.compile(r",?\s*rotate\s*=\s*0(?:\.0)?(?=\s*[,\]])\b"),
    re.compile(r",?\s*solid(?=\s*[,\]])\b"),
)


def strip_default_options(text: str) -> str:
    """Remove TikZ style options that equal the renderer default and thus have no effect.

    For example ``\\draw[draw=black, line width=0.4pt]`` becomes ``\\draw`` because
    those are both TikZ defaults.  This saves 2–5 tokens per affected command and
    prevents the model from learning unnecessary verbosity.

    Empty option brackets are cleaned up afterwards: ``[]`` → removed.
    """
    for pattern in _DEFAULT_OPTION_PATTERNS:
        text = pattern.sub("", text)
    # Clean up brackets that became empty or contain only whitespace/commas.
    text = re.sub(r"\[\s*,?\s*\]", "", text)
    text = re.sub(r"\[\s*,\s*", "[", text)  # leading comma inside bracket
    text = re.sub(r",\s*\]", "]", text)     # trailing comma inside bracket
    # Compress padding whitespace inside brackets to save tokens
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)
    # Compress coordinate whitespace
    text = re.sub(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", r"(\1,\2)", text)
    return text


def quantize_floats(text: str, decimals: int = 3) -> str:
    """Round all floating-point numbers to *decimals* significant decimal places.

    The model has no ability to learn differences at the 9th decimal place; those
    extra digits only waste token budget and inject spurious gradient signal.
    Only floats with *more* than ``decimals`` decimal digits are touched.  Integer
    values and already-short floats are left untouched.

    Examples::

        0.941517254116162  →  0.942   (saves ~13 tokens)
        -1.20711           →  -1.207
        2.5                →  2.5     (unchanged)
        3                  →  3       (unchanged)
    """
    # Only target floats that have strictly more than `decimals` decimal places.
    pattern = re.compile(r"-?\d+\.\d{" + str(decimals + 1) + r",}")

    def _round_match(m: re.Match) -> str:  # type: ignore[type-arg]
        val = float(m.group(0))
        rounded = f"{val:.{decimals}f}"
        # Strip trailing zeros but keep at least one decimal if the value is
        # non-integer, so 1.000 → 1.0 (not 1), preserving float context.
        if "." in rounded:
            rounded = rounded.rstrip("0")  # 1.200 → 1.2
            if rounded.endswith("."):      # 2. → 2.0
                rounded += "0"
        return rounded

    return pattern.sub(_round_match, text)


def derep_duplicate_commands(text: str) -> str:
    """Remove repeated identical TikZ draw/node/path command lines (DeRep §statement-level).

    Keeps the first occurrence of any duplicate \\draw, \\node, \\path, \\fill,
    \\coordinate, or \\addplot line. Lines inside preamble (\\usepackage, \\documentclass,
    \\usetikzlibrary, etc.) are left untouched to avoid corrupting the document structure.

    Only self-contained lines (where braces are balanced on the line itself) are
    eligible for deduplication.  Lines like ``\\addplot coordinates{`` have an
    opening brace whose matching ``}`` lives on a later line; removing such a
    line would break brace balance.

    This matches Extended §4.2 (DeRep statement-level repetition removal).
    """
    COMMAND_PREFIX_RE = re.compile(
        r"^\s*\\(?:draw|fill|filldraw|node|path|coordinate|addplot|clip)\b",
    )
    seen: set[str] = set()
    result_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if COMMAND_PREFIX_RE.match(stripped):
            # Only deduplicate lines whose braces are self-contained.
            # If opens != closes, this line is part of a multi-line command
            # and removing it would unbalance the surrounding braces.
            if stripped.count("{") == stripped.count("}"):
                if stripped in seen:
                    continue  # drop duplicate
                seen.add(stripped)
        result_lines.append(line)
    return "\n".join(result_lines)



def normalize_tikz(text: str) -> str:
    """Apply the full multi-pass normalization and healing pipeline to a TikZ string.

    This includes comment stripping, dependency removal, environment healing,
    wrapping in a standalone document class for consistent compilation,
    float quantization to cap coordinate precision, and DeRep statement-level
    de-duplication of repeated draw commands.
    """
    cleaned = strip_inline_comments(text)
    cleaned = strip_external_dependency_lines(cleaned)
    cleaned = ensure_standalone_document(cleaned)
    cleaned = strip_default_options(cleaned)
    cleaned = quantize_floats(cleaned)
    return derep_duplicate_commands(cleaned)


def extract_primary_environment(text: str) -> str | None:
    for environment in ("tikzpicture", "tikz-cd", "circuitikz"):
        if re.search(rf"\\begin\{{{re.escape(environment)}\}}", text):
            return environment
    return None
