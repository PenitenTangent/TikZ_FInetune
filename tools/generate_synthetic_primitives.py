import json
import hashlib
import sys
from pathlib import Path

# Add src to pythonpath so we can import prompting
sys.path.append(str(Path(__file__).parent.parent / "src"))
from tikz_mlx.prompting import build_generation_prompt, PROMPT_CONTRACT_VERSION, prompt_template_sha256

SEED_PRIMITIVES = [
    (
        "Draw a simple black line from the origin (0,0) to coordinate (1,1).",
        "\\begin{tikzpicture}\n\\draw (0,0) -- (1,1);\n\\end{tikzpicture}"
    ),
    (
        "Draw a red circle centered at (0,0) with a radius of 1.",
        "\\begin{tikzpicture}\n\\draw[red] (0,0) circle (1);\n\\end{tikzpicture}"
    ),
    (
        "Draw a blue rectangle with bottom-left corner at (0,0) and top-right corner at (2,1).",
        "\\begin{tikzpicture}\n\\draw[blue] (0,0) rectangle (2,1);\n\\end{tikzpicture}"
    ),
    (
        "Draw a sine wave from x=0 to x=6 using a smooth plot.",
        "\\begin{tikzpicture}\n\\draw domain=0:6,smooth plot (\\x, {sin(\\x r)});\n\\end{tikzpicture}"
    ),
    (
        "Draw a black arrow pointing from (0,0) to (2,0).",
        "\\begin{tikzpicture}\n\\draw[->] (0,0) -- (2,0);\n\\end{tikzpicture}"
    ),
    (
        "Create a text node at coordinate (1,1) containing the word 'Hello'.",
        "\\begin{tikzpicture}\n\\node at (1,1) {Hello};\n\\end{tikzpicture}"
    ),
    (
        "Draw a grid from (-2,-2) to (2,2) with thin gray lines.",
        "\\begin{tikzpicture}\n\\draw[help lines, thin, gray] (-2,-2) grid (2,2);\n\\end{tikzpicture}"
    ),
    (
        "Fill a unit square from (0,0) to (1,1) with blue color.",
        "\\begin{tikzpicture}\n\\fill[blue] (0,0) rectangle (1,1);\n\\end{tikzpicture}"
    ),
    (
        "Draw a coordinate axis with an x-axis from -1 to 3 and a y-axis from -1 to 3, both with arrows.",
        "\\begin{tikzpicture}\n\\draw[->] (-1,0) -- (3,0);\n\\draw[->] (0,-1) -- (0,3);\n\\end{tikzpicture}"
    ),
    (
        "Draw a green ellipse centered at (1,1) with x-radius 2 and y-radius 1.",
        "\\begin{tikzpicture}\n\\draw[green] (1,1) ellipse (2 and 1);\n\\end{tikzpicture}"
    ),
    (
        "Draw a dashed line from (0,1) to (3,1).",
        "\\begin{tikzpicture}\n\\draw[dashed] (0,1) -- (3,1);\n\\end{tikzpicture}"
    ),
    (
        "Draw a thick red line connecting (0,0), (1,2), and (2,0).",
        "\\begin{tikzpicture}\n\\draw[thick, red] (0,0) -- (1,2) -- (2,0);\n\\end{tikzpicture}"
    ),
    (
        "Draw a solid black triangle with vertices at (0,0), (2,0), and (1,1.732).",
        "\\begin{tikzpicture}\n\\fill (0,0) -- (2,0) -- (1,1.732) -- cycle;\n\\end{tikzpicture}"
    ),
    (
        "Draw a circular node at (0,0) with a blue border containing the number 1.",
        "\\begin{tikzpicture}\n\\node[draw=blue, circle] at (0,0) {1};\n\\end{tikzpicture}"
    ),
    (
        "Draw a parabola from (-2,4) to (2,4) with its vertex at (0,0).",
        "\\begin{tikzpicture}\n\\draw (-2,4) parabola bend (0,0) (2,4);\n\\end{tikzpicture}"
    ),
    (
        "Draw a rectangle with rounded corners from (0,0) to (3,2).",
        "\\begin{tikzpicture}\n\\draw[rounded corners] (0,0) rectangle (3,2);\n\\end{tikzpicture}"
    ),
    (
        "Create a red text node at (0,0) saying 'Warning'.",
        "\\begin{tikzpicture}\n\\node[text=red] at (0,0) {Warning};\n\\end{tikzpicture}"
    ),
    (
        "Draw a straight line from node A to node B.",
        "\\begin{tikzpicture}\n\\node (A) at (0,0) {A};\n\\node (B) at (2,0) {B};\n\\draw (A) -- (B);\n\\end{tikzpicture}"
    ),
    (
        "Draw a horizontal dashed line from x=-2 to x=2 along the y-axis (y=0).",
        "\\begin{tikzpicture}\n\\draw[dashed] (-2,0) -- (2,0);\n\\end{tikzpicture}"
    ),
    (
        "Draw a cyan filled circle centered at (2,2) with radius 0.5.",
        "\\begin{tikzpicture}\n\\fill[cyan] (2,2) circle (0.5);\n\\end{tikzpicture}"
    )
]


def _sample_id(index: int, description: str, code: str) -> str:
    return hashlib.sha256(f"{index}\n{description}\n{code}".encode("utf-8")).hexdigest()[:16]


def _fmt(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def build_primitives() -> list[tuple[str, str]]:
    """Build a deterministic but varied Stage 0 warmup set.

    The old file repeated 20 primitives 50 times. After eval decontamination,
    that collapsed to a very small repeated subset, which made the warmup
    adapter memorize prompt-contract words instead of learning TikZ starts.
    """
    primitives: list[tuple[str, str]] = list(SEED_PRIMITIVES)
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "teal", "gray", "black"]
    styles = ["solid", "dashed", "dotted", "thick", "very thick"]

    for i in range(80):
        x1 = (i % 9) - 4
        y1 = ((i * 2) % 7) - 3
        x2 = x1 + 1 + (i % 4)
        y2 = y1 + ((i * 3) % 5) - 2
        color = colors[i % len(colors)]
        style = styles[i % len(styles)]
        primitives.append((
            f"Draw a {style} {color} segment from ({_fmt(x1)},{_fmt(y1)}) to ({_fmt(x2)},{_fmt(y2)}).",
            "\\begin{tikzpicture}\n"
            f"\\draw[{style}, {color}] ({_fmt(x1)},{_fmt(y1)}) -- ({_fmt(x2)},{_fmt(y2)});\n"
            "\\end{tikzpicture}",
        ))

    for i in range(70):
        cx = ((i * 2) % 11) - 5
        cy = ((i * 3) % 9) - 4
        radius = 0.3 + (i % 6) * 0.2
        color = colors[(i + 2) % len(colors)]
        fill = i % 3 == 0
        verb = "filled" if fill else "outlined"
        command = "\\fill" if fill else "\\draw"
        primitives.append((
            f"Create a {verb} {color} circle centered at ({_fmt(cx)},{_fmt(cy)}) with radius {_fmt(radius)}.",
            "\\begin{tikzpicture}\n"
            f"{command}[{color}] ({_fmt(cx)},{_fmt(cy)}) circle ({_fmt(radius)});\n"
            "\\end{tikzpicture}",
        ))

    for i in range(70):
        x = (i % 8) - 4
        y = ((i * 3) % 8) - 4
        w = 1 + (i % 4) * 0.5
        h = 0.6 + (i % 5) * 0.4
        color = colors[(i + 4) % len(colors)]
        rounded = i % 4 == 0
        option = f"{color}, rounded corners" if rounded else color
        primitives.append((
            f"Draw a {color} rectangle from ({_fmt(x)},{_fmt(y)}) to ({_fmt(x + w)},{_fmt(y + h)})"
            + (" with rounded corners." if rounded else "."),
            "\\begin{tikzpicture}\n"
            f"\\draw[{option}] ({_fmt(x)},{_fmt(y)}) rectangle ({_fmt(x + w)},{_fmt(y + h)});\n"
            "\\end{tikzpicture}",
        ))

    for i in range(60):
        cx = (i % 7) - 3
        cy = ((i * 4) % 7) - 3
        rx = 0.8 + (i % 5) * 0.3
        ry = 0.4 + (i % 4) * 0.2
        color = colors[(i + 6) % len(colors)]
        primitives.append((
            f"Draw a {color} ellipse centered at ({_fmt(cx)},{_fmt(cy)}) with x-radius {_fmt(rx)} and y-radius {_fmt(ry)}.",
            "\\begin{tikzpicture}\n"
            f"\\draw[{color}] ({_fmt(cx)},{_fmt(cy)}) ellipse ({_fmt(rx)} and {_fmt(ry)});\n"
            "\\end{tikzpicture}",
        ))

    for i in range(60):
        x = (i % 6) - 3
        y = ((i * 5) % 6) - 3
        text = f"P{i:02d}"
        color = colors[(i + 1) % len(colors)]
        shape = "circle" if i % 2 == 0 else "rectangle"
        primitives.append((
            f"Place a {shape} node labeled {text} at ({_fmt(x)},{_fmt(y)}) with a {color} outline.",
            "\\begin{tikzpicture}\n"
            f"\\node[draw={color}, {shape}] at ({_fmt(x)},{_fmt(y)}) {{{text}}};\n"
            "\\end{tikzpicture}",
        ))

    for i in range(60):
        x1 = (i % 7) - 3
        y1 = ((i * 2) % 7) - 3
        x2 = x1 + 1.0 + (i % 4) * 0.5
        y2 = y1 + 0.8 + (i % 3) * 0.4
        color = colors[(i + 3) % len(colors)]
        primitives.append((
            f"Fill a {color} triangle with corners ({_fmt(x1)},{_fmt(y1)}), ({_fmt(x2)},{_fmt(y1)}), and ({_fmt((x1 + x2) / 2)},{_fmt(y2)}).",
            "\\begin{tikzpicture}\n"
            f"\\fill[{color}] ({_fmt(x1)},{_fmt(y1)}) -- ({_fmt(x2)},{_fmt(y1)}) -- ({_fmt((x1 + x2) / 2)},{_fmt(y2)}) -- cycle;\n"
            "\\end{tikzpicture}",
        ))

    for i in range(40):
        xmin = -1 - (i % 4)
        ymin = -1 - ((i + 1) % 4)
        xmax = 1 + ((i + 2) % 4)
        ymax = 1 + ((i + 3) % 4)
        primitives.append((
            f"Draw a light gray helper grid spanning ({xmin},{ymin}) to ({xmax},{ymax}).",
            "\\begin{tikzpicture}\n"
            f"\\draw[help lines, gray] ({xmin},{ymin}) grid ({xmax},{ymax});\n"
            "\\end{tikzpicture}",
        ))

    return primitives

def main():
    output_path = Path("data/prepared/curriculum/synthetic_primitives.jsonl")

    primitives = build_primitives()
    total_examples = len(primitives)

    print(f"Generating {total_examples} synthetic primitive examples...")

    template_hash = prompt_template_sha256()

    with open(output_path, "w") as f:
        for idx, (desc, code) in enumerate(primitives):
            prompt = build_generation_prompt(desc, generation_mode="plain_tikz")
            assistant_message = f"{code}\n```"
            record = {
                "example_index": idx,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_message},
                ],
                "metadata": {
                    "example_index": idx,
                    "generation_mode": "plain_tikz",
                    "prompt_contract_version": PROMPT_CONTRACT_VERSION,
                    "prompt_template_sha256": template_hash,
                    "target_contract": "body_only_environment",
                },
                "sample_id": _sample_id(idx, desc, code),
                "source": "synthetic:primitives",
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Saved to {output_path}")
    print("\nTo use this, you should prepend it to your training data:")
    print("cat data/prepared/curriculum/synthetic_primitives.jsonl data/prepared/curriculum/train_stage1_clean.jsonl > data/prepared/curriculum/train_stage1_anchored.jsonl")
    print("Then update your config to point to train_stage1_anchored.jsonl")

if __name__ == "__main__":
    main()
