from tikz_mlx.static_critic import analyze_tikz_static


def test_static_critic_reports_basic_features() -> None:
    report = analyze_tikz_static(
        "```latex\n\\begin{document}\\begin{tikzpicture}\\draw (0,0) -- (1.2345,1);\\end{tikzpicture}\\end{document}\n```",
        generation_mode="plain_tikz",
    )

    assert report.fence_present is True
    assert report.has_begin_document is True
    assert report.has_tikzpicture is True
    assert report.num_high_precision_floats == 1
    assert report.brace_balance == 0
    assert report.violations == []


def test_static_critic_flags_cross_dialect_pgfplots_keys() -> None:
    report = analyze_tikz_static(
        "\\begin{tikzpicture}\\draw[axis x line*=bottom] (0,0) -- (1,1);\\end{tikzpicture}",
        generation_mode="plain_tikz",
    )

    assert report.has_pgfplots_keys is True
    assert "pgfplots_keys_in_plain_tikz" in report.violations


def test_static_critic_allows_plain_tikz_grid_path_operation() -> None:
    report = analyze_tikz_static(
        "\\begin{tikzpicture}\\draw (0,0) grid (2,2);\\end{tikzpicture}",
        generation_mode="plain_tikz",
    )

    assert report.has_pgfplots_keys is False
    assert "pgfplots_keys_in_plain_tikz" not in report.violations


def test_static_critic_flags_legacy_style_and_missing_axis() -> None:
    report = analyze_tikz_static(
        "\\tikzstyle{vertex}=[circle,draw]\\begin{tikzpicture}\\draw (0,0);\\end{tikzpicture}",
        generation_mode="pgfplots_axis",
    )

    assert report.has_legacy_tikzstyle is True
    assert "legacy_tikzstyle" in report.violations
    assert "missing_axis_environment" in report.violations
