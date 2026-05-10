r"""grammar.py — Grammar-guided streaming state machine for TikZ generation.

Implements plan Extended §4 "Grammar-Aware Decoding" within the constraints of
mlx-vlm's streaming API (text chunks, no logit access).

The `TikzGrammarState` class is a lightweight finite state machine that is fed
streamed text chunks during generation and signals when to abort early:

  - **Environment balance**: tracks ``\begin{E}`` / ``\end{E}`` counts per
    environment type. Aborts if a primary environment (tikzpicture / axis /
    circuitikz / tikz-cd) is opened a second time while one is already open
    (duplicate-open = hallucination of a nested diagram).

  - **Post-closure pollution**: once the primary environment has been closed
    (``\end{tikzpicture}`` etc.), any subsequent non-whitespace content is a
    sign the model is hallucinating extra code. After a short grace window (to
    allow ``\end{document}`` and the closing fence), abort.

  - **Runaway depth**: if brace depth or bracket depth exceeds a hard limit,
    abort (prevents exponential nesting hallucinations).

  - **Fence tracking**: correctly terminates on the second ```` ``` ```` marker.

Usage::

    state = TikzGrammarState()
    for chunk in stream_generate(...):
        text += chunk
        state.feed(chunk)
        if state.should_abort:
            break  # grammar-triggered early exit
"""

from __future__ import annotations

import re

# Primary TikZ environments — the document body lives inside exactly one of these.
_PRIMARY_ENVS: tuple[str, ...] = (
    "tikzpicture",
    "axis",
    "semilogxaxis",
    "semilogyaxis",
    "circuitikz",
    "tikzcd",
    "tikz-cd",
)

# Secondary environments that nest inside a primary — not abort triggers.
_SECONDARY_ENVS: frozenset[str] = frozenset({
    "scope",
    "pgfonlayer",
    "groupplot",
    "pgfinterruptboundingbox",
})

# Maximum allowed brace / bracket depth before we assume runaway nesting.
_MAX_BRACE_DEPTH = 30
_MAX_BRACKET_DEPTH = 20

# How many non-whitespace characters after the primary environment closes are
# permitted before we abort (gives room for \end{document} + ``` + newline).
_POST_CLOSE_GRACE_CHARS = 60

_BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
_END_RE = re.compile(r"\\end\{([^}]+)\}")
# \foreach \var in {items}: capture the items group for range analysis.
_FOREACH_RE = re.compile(r"\\foreach\s+\\[a-zA-Z]+\s+in\s*\{([^}]*)\}")
_MAX_FOREACH_ITERATIONS = 500   # single loop range limit
_MAX_FOREACH_NESTING   = 4      # max nested \foreach depth


class TikzGrammarState:
    """Streaming grammar tracker for TikZ generation.

    Feed successive text chunks via :meth:`feed`. Check :attr:`should_abort`
    after each chunk.  :attr:`abort_reason` contains a human-readable
    description of why the abort was triggered.
    """

    __slots__ = (
        "_primary_open",        # name of the primary env currently open, or None
        "_primary_open_count",  # how many times the current primary was opened
        "_env_depths",          # {env_name: open_count} for all envs
        "_primary_closed",      # True once the primary env has been fully closed
        "_post_close_nonws",    # non-whitespace char count after primary closed
        "_brace_depth",         # running { } depth
        "_bracket_depth",       # running [ ] depth
        "_fence_count",         # number of ``` markers seen
        "_accumulated",         # full accumulated text (for regex matching)
        "_foreach_depth",       # current nesting depth of \foreach
        "should_abort",         # public flag
        "abort_reason",         # human-readable reason
    )

    def __init__(self) -> None:
        self._primary_open: str | None = None
        self._primary_open_count: int = 0
        self._env_depths: dict[str, int] = {}
        self._primary_closed: bool = False
        self._post_close_nonws: int = 0
        self._brace_depth: int = 0
        self._bracket_depth: int = 0
        self._fence_count: int = 0
        self._accumulated: str = ""
        self._foreach_depth: int = 0
        self.should_abort: bool = False
        self.abort_reason: str = ""

    # ------------------------------------------------------------------ public

    def feed(self, chunk: str) -> None:
        """Process a new streamed text chunk.

        Updates internal state and sets :attr:`should_abort` if a grammar
        violation is detected.  Safe to call after abort (no-op).
        """
        if self.should_abort:
            return

        self._accumulated += chunk
        self._process_chunk(chunk)

    # ----------------------------------------------------------------- private

    def _process_chunk(self, chunk: str) -> None:  # noqa: C901
        i = 0
        n = len(chunk)
        while i < n:
            c = chunk[i]

            # ── TeX comment: skip to end of line ──────────────────────────────────
            # An unescaped % starts a TeX comment; all content until the next
            # newline is ignored so we never misparse `% \begin{tikzpicture}`.
            if c == "%" and (i == 0 or chunk[i - 1] != "\\"):
                while i < n and chunk[i] != "\n":
                    i += 1
                continue

            # ── fence detection ───────────────────────────────────────────────
            if chunk[i:i + 3] == "```":
                self._fence_count += 1
                if self._fence_count >= 2:
                    # Second fence = closing code block — stop is correct but NOT
                    # an error; the generate loop already handles this as a stop
                    # token. We don't set should_abort here.
                    return
                i += 3
                continue

            # ── brace depth ───────────────────────────────────────────────────
            if c == "{":
                self._brace_depth += 1
                if self._brace_depth > _MAX_BRACE_DEPTH:
                    self._abort(f"brace depth exceeded {_MAX_BRACE_DEPTH}")
                    return
            elif c == "}":
                self._brace_depth = max(0, self._brace_depth - 1)

            # ── bracket depth ─────────────────────────────────────────────────
            elif c == "[":
                self._bracket_depth += 1
                if self._bracket_depth > _MAX_BRACKET_DEPTH:
                    self._abort(f"bracket depth exceeded {_MAX_BRACKET_DEPTH}")
                    return
            elif c == "]":
                self._bracket_depth = max(0, self._bracket_depth - 1)

            # ── environment and \foreach tracking ────────────────────────────
            elif c == "\\":
                # Try to match \begin{...} or \end{...} at this position.
                rest = chunk[i:]
                begin_m = _BEGIN_RE.match(rest)
                if begin_m:
                    env = begin_m.group(1).strip()
                    self._handle_begin(env)
                    if self.should_abort:
                        return
                    i += begin_m.end()
                    continue
                end_m = _END_RE.match(rest)
                if end_m:
                    env = end_m.group(1).strip()
                    self._handle_end(env)
                    if self.should_abort:
                        return
                    i += end_m.end()
                    continue
                # Try to match \foreach \var in {items}.
                foreach_m = _FOREACH_RE.match(rest)
                if foreach_m:
                    items_str = foreach_m.group(1).strip()
                    # Detect numeric range syntax: {start,...,end}
                    range_m = re.match(r"(-?\d+)(?:\s*,\s*-?\d+)?\s*,\s*\.\.\.\s*,\s*(-?\d+)", items_str)
                    if range_m:
                        try:
                            n_iters = abs(int(range_m.group(2)) - int(range_m.group(1))) + 1
                            if n_iters > _MAX_FOREACH_ITERATIONS:
                                self._abort(
                                    f"\\foreach range too large ({n_iters} iterations > {_MAX_FOREACH_ITERATIONS})"
                                )
                                return
                        except ValueError:
                            pass
                    self._foreach_depth += 1
                    if self._foreach_depth > _MAX_FOREACH_NESTING:
                        self._abort(
                            f"\\foreach nesting exceeded {_MAX_FOREACH_NESTING} levels"
                        )
                        return
                    i += foreach_m.end()
                    continue

            # ── post-close pollution check ────────────────────────────────────
            if self._primary_closed and not c.isspace():
                self._post_close_nonws += 1
                if self._post_close_nonws > _POST_CLOSE_GRACE_CHARS:
                    self._abort(
                        f"content after primary environment closed "
                        f"({self._post_close_nonws} non-ws chars)"
                    )
                    return

            i += 1

    def _handle_begin(self, env: str) -> None:
        env_norm = env.lower()
        depth = self._env_depths.get(env_norm, 0) + 1
        self._env_depths[env_norm] = depth

        if env_norm in _PRIMARY_ENVS:
            if self._primary_open is None:
                # First opening of any primary env — normal.
                self._primary_open = env_norm
                self._primary_open_count = 1
            elif env_norm == self._primary_open:
                # Same primary env opened again while one is already open.
                self._primary_open_count += 1
                if self._primary_open_count > 1:
                    self._abort(
                        f"duplicate \\begin{{{env}}} inside open {self._primary_open} environment"
                    )
            else:
                # Different primary env opened while another is open.
                self._abort(
                    f"\\begin{{{env}}} opened while {self._primary_open} is already open"
                )

    def _handle_end(self, env: str) -> None:
        env_norm = env.lower()
        depth = max(0, self._env_depths.get(env_norm, 0) - 1)
        self._env_depths[env_norm] = depth

        if env_norm in _PRIMARY_ENVS and env_norm == self._primary_open:
            self._primary_open_count -= 1
            if self._primary_open_count <= 0:
                self._primary_closed = True
                self._post_close_nonws = 0

    def _abort(self, reason: str) -> None:
        self.should_abort = True
        self.abort_reason = reason
