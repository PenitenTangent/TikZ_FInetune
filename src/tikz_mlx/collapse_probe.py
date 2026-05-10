import re
import mlx.core as mx
from mlx_vlm.generate import generate

SENTINEL_PROMPTS = [
    "Generate a vertical black line.",
    "Draw a red circle centered at (0,0).",
    "Create a simple flow chart with two boxes.",
    "Draw a sine wave from x=0 to x=6."
]

def check_for_collapse(text: str) -> list[str]:
    reasons = []
    if "\\PreviewEnvironment" in text:
        reasons.append("Contains \\PreviewEnvironment")
    if "\\usepackage" in text:
        reasons.append("Contains \\usepackage")
    if "\\documentclass" in text:
        reasons.append("Contains \\documentclass")
    
    # Check for repetition loops (e.g. same line 5 times)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for i in range(len(lines) - 5):
        if len(set(lines[i:i+5])) == 1:
            reasons.append("Repetition loop detected")
            break
            
    # Check for excessive length
    if len(text) > 4000:
        reasons.append(f"Excessive length ({len(text)} chars)")
        
    return reasons

def run_collapse_probe(model, processor, build_prompt_fn, verbose=False):
    """
    Runs a fast probe on the model to see if it has collapsed.
    Returns (passed, list of failures)
    """
    model.eval()
    failures = []
    
    for prompt_text in SENTINEL_PROMPTS:
        # We need to build the actual prompt using the contract
        full_prompt = build_prompt_fn(prompt_text)
        
        try:
            # Generate a small sample
            result = generate(
                model=model,
                processor=processor,
                prompt=full_prompt,
                max_tokens=512,
                verbose=False
            )
            text = result.text
            reasons = check_for_collapse(text)
            if reasons:
                failures.append({"prompt": prompt_text, "response": text, "reasons": reasons})
        except Exception as e:
            if verbose:
                print(f"Probe failed for prompt '{prompt_text}': {e}")
            continue
            
    model.train()
    return len(failures) == 0, failures
