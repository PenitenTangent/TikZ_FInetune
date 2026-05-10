from tikz_mlx.compiler import parse_tectonic_log


def test_parse_tectonic_log_extracts_errors_hints_and_packages() -> None:
    log_text = """
! LaTeX Error: File `foo.sty' not found.
l.12 \\usepackage{foo}
! Package tikz Error: Giving up on this path. Did you forget a semicolon?.
l.20 \\draw (0,0) -- (1,1)
"""
    errors, line_hints, missing_packages = parse_tectonic_log(log_text)
    assert "LaTeX Error: File `foo.sty' not found." in errors
    assert "Package tikz Error: Giving up on this path. Did you forget a semicolon?." in errors
    assert line_hints == [12, 20]
    assert missing_packages == ["foo.sty"]
