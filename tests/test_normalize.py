from tikz_mlx.normalize import (
    contains_external_dependencies,
    detect_required_packages,
    detect_required_tikz_libraries,
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


def test_detect_required_packages_adds_plot_and_circuit_packages() -> None:
    packages = detect_required_packages(
        r"\begin{axis}\addplot coordinates {(0,0) (1,1)};\end{axis}"
        "\n"
        r"\begin{circuitikz}\draw (0,0) to[R] (1,0);\end{circuitikz}"
    )

    assert r"\usepackage{pgfplots}" in packages
    assert r"\pgfplotsset{compat=1.18}" in packages
    assert r"\usepackage{circuitikz}" in packages


def test_detect_required_tikz_libraries_from_syntax() -> None:
    raw = (
        r"\begin{tikzpicture}"
        r"\node[right=of a] (b) {B};"
        r"\draw[-Stealth,decorate,decoration={brace}] ($(a)+(1,0)$) -- (b);"
        r"\matrix[matrix of nodes] {A & B\\};"
        r"\path[name intersections={of=a and b}];"
        r"\end{tikzpicture}"
    )

    libraries = detect_required_tikz_libraries(raw)

    assert "arrows.meta" in libraries
    assert "calc" in libraries
    assert "decorations.pathreplacing" in libraries
    assert "intersections" in libraries
    assert "matrix" in libraries
    assert "positioning" in libraries


def test_detect_required_packages_preserves_explicit_packages_and_libraries() -> None:
    normalized = normalize_tikz(
        r"\documentclass{article}"
        "\n"
        r"\usepackage{siunitx}"
        "\n"
        r"\usetikzlibrary{patterns, fit}"
        "\n"
        r"\begin{document}\begin{tikzpicture}\node[fit=(a)] {};\end{tikzpicture}\end{document}"
    )

    assert r"\usepackage{siunitx}" in normalized
    assert r"\usetikzlibrary{fit, patterns}" in normalized


def test_external_dependency_detection() -> None:
    raw = r"\includegraphics{figure.png}"
    assert contains_external_dependencies(raw) is True
