from __future__ import annotations

import re

from .schemas import ExtractedBlock

TARGET_ENVIRONMENTS = ("tikzpicture", "tikz-cd", "circuitikz", "subfigure")
TOKEN_RE = re.compile(r"\\(begin|end)\{(" + "|".join(re.escape(name) for name in TARGET_ENVIRONMENTS) + r")\}")


def extract_environments(text: str, target_environments: tuple[str, ...] = ("tikzpicture", "tikz-cd", "circuitikz")) -> list[ExtractedBlock]:
    stack: list[tuple[str, int]] = []
    blocks: list[ExtractedBlock] = []

    for match in TOKEN_RE.finditer(text):
        kind, environment = match.group(1), match.group(2)
        if kind == "begin":
            stack.append((environment, match.start()))
            continue

        for index in range(len(stack) - 1, -1, -1):
            current_env, start = stack[index]
            if current_env != environment:
                continue

            stack = stack[:index]
            if environment in target_environments:
                parent_environment = stack[-1][0] if stack else None
                blocks.append(
                    ExtractedBlock(
                        environment=environment,
                        text=text[start : match.end()],
                        start=start,
                        end=match.end(),
                        parent_environment=parent_environment,
                    )
                )
            break

    return sorted(blocks, key=lambda block: block.start)


def extract_tikz_blocks(text: str) -> list[ExtractedBlock]:
    subfigures = [block for block in extract_environments(text, ("subfigure",)) if block.environment == "subfigure"]
    if not subfigures:
        return extract_environments(text)

    extracted: list[ExtractedBlock] = []
    for subfigure in subfigures:
        extracted.extend(extract_environments(subfigure.text))
    return extracted or extract_environments(text)
