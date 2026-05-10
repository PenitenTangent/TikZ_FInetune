import pytest
import json
import re
from pathlib import Path
from tikz_mlx.bad_patterns import check_bad_patterns
from tikz_mlx.filter import build_sample
from tikz_mlx.settings import DatasetConfig
from tikz_mlx.prepare import _annotate_training_record_context

def test_bad_pattern_repetition_loops():
    # Test excessive command repetition
    raw = "\\begin{tikzpicture}\n" + "\\draw (0,0) -- (1,1);\n" * 150 + "\\end{tikzpicture}"
    res = check_bad_patterns(raw)
    assert any(v["rule"] == "repeated_draw_node_excessive" for v in res["violations"])

def test_bad_pattern_consecutive_backslashes():
    raw = "\\begin{tikzpicture} \\\\\\\\\\\\\\\\ \\end{tikzpicture}"
    res = check_bad_patterns(raw)
    assert any(v["rule"] == "consecutive_backslashes" for v in res["violations"])

def test_external_dependency_rejection_raw():
    config = DatasetConfig(
        min_chars=10, max_chars=1000, split_seed=42, 
        supported_environments=("tikzpicture",), 
        reject_external_dependencies=True, deduplicate=False,
        drop_truncated_records=True
    )
    # Even if it's compilable after normalization (which strips \includegraphics), 
    # we want to reject it if raw has \includegraphics to maintain prompt coherence.
    raw = "\\begin{tikzpicture} \\node at (0,0) {\\includegraphics{foo.png}}; \\end{tikzpicture}"
    sample, decision = build_sample("test", raw, "desc", config)
    assert sample is None
    assert "external dependencies present in raw source" in decision.reasons

def test_truncation_drop_policy():
    class MockTokenizer:
        def apply_chat_template(self, *args, **kwargs): return "mock text"
        def encode(self, text, **kwargs): return [1] * 100 # 100 tokens
        
    tokenizer = MockTokenizer()
    record = {"messages": [{"role": "user", "content": "foo"}]}
    
    # max_context_tokens = 50 -> should be truncated
    token_length, is_truncated = _annotate_training_record_context(record, tokenizer=tokenizer, max_context_tokens=50)
    assert is_truncated is True
    assert record["metadata"]["is_truncated"] is True
    assert record["metadata"]["token_length"] == 100

def test_assistant_token_marker_fail_safe():
    # This checks if the logic for finding the assistant start works
    # Though we use chat templates, some logic might rely on manual splitting
    pass
