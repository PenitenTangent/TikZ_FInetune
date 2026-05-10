from tikz_mlx.normalize import (
    contains_external_dependencies,
    detect_required_packages,
    normalize_tikz,
    strip_inline_comments,
)


def test_strip_inline_comments_preserves_escaped_percent() -> None:
    text = r"\draw (0,0) -- (1,1); % remove this" + "\n" + r"\node {100\%};"
    cleaned = strip_inline_comments(text)
    assert "% remove this" not in cleaned
    assert r"100\%" in cleaned


def test_normalize_wraps_standalone_document() -> None:
    raw = r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}"
    normalized = normalize_tikz(raw)
    assert r"\documentclass{article}" in normalized
    assert r"\usepackage[active,tightpage]{preview}" in normalized
    assert r"\begin{document}" in normalized
    assert r"\end{document}" in normalized


def test_detect_required_packages_adds_tikz_cd() -> None:
    raw = r"\begin{tikz-cd} A \arrow[r] & B \end{tikz-cd}"
    packages = detect_required_packages(raw)
    assert r"\usepackage{tikz}" in packages
    assert r"\usepackage{tikz-cd}" in packages


def test_external_dependency_detection() -> None:
    raw = r"\includegraphics{figure.png}"
    assert contains_external_dependencies(raw) is True
