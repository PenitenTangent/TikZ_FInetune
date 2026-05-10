from tikz_mlx.extract import extract_tikz_blocks


def test_extract_tikz_blocks_from_document() -> None:
    text = r"""
    \begin{figure}
    \begin{tikzpicture}
    \draw (0,0) -- (1,1);
    \end{tikzpicture}
    \end{figure}
    """
    blocks = extract_tikz_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].environment == "tikzpicture"


def test_extract_prefers_subfigure_contents_when_present() -> None:
    text = r"""
    \begin{subfigure}
    \begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}
    \end{subfigure}
    \begin{subfigure}
    \begin{tikz-cd} A \arrow[r] & B \end{tikz-cd}
    \end{subfigure}
    """
    blocks = extract_tikz_blocks(text)
    assert len(blocks) == 2
    assert {block.environment for block in blocks} == {"tikzpicture", "tikz-cd"}
