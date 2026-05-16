import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import mlx_lm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from tikz_mlx.prompting import build_generation_prompt

PROMPTS = [
    "Draw a red circle centered at the origin with radius 2.",
    "Draw a grid from (-2,-2) to (2,2) with thin gray lines.",
    "Draw a simple black line from the origin (0,0) to coordinate (1,1).",
    "Draw a blue triangle with vertices at (0,0), (2,0), and (1,2).",
    "Draw a coordinate system with arrows for x and y axes, ranging from -1 to 5."
]

def format_prompt(desc):
    return build_generation_prompt(desc, generation_mode="plain_tikz")

def test_llama_cpp(model_id, filename):
    print(f"\n=========================================")
    print(f"Testing {model_id} (Fine-tuned)")
    print(f"=========================================")
    
    path = hf_hub_download(repo_id=model_id, filename=filename)
    # Using low temperature for deterministic testing
    llm = Llama(model_path=path, n_ctx=2048, verbose=False)
    
    for desc in PROMPTS:
        prompt = format_prompt(desc)
        print(f"\n--- PROMPT: {desc} ---")
        
        chat = [{"role": "user", "content": prompt}]
        
        t0 = time.time()
        res = llm.create_chat_completion(
            messages=chat, 
            max_tokens=512,
            temperature=0.1
        )
        t1 = time.time()
        
        content = res["choices"][0]["message"]["content"]
        print(content)
        print(f"[Took {t1-t0:.2f}s]")

def test_mlx(model_id):
    print(f"\n=========================================")
    print(f"Testing {model_id} (Base Gemma)")
    print(f"=========================================")
    
    model, tokenizer = mlx_lm.load(model_id)
    
    for desc in PROMPTS:
        prompt = format_prompt(desc)
        print(f"\n--- PROMPT: {desc} ---")
        
        chat = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        t0 = time.time()
        res = mlx_lm.generate(
            model, 
            tokenizer, 
            prompt=text, 
            max_tokens=512, 
            verbose=False
        )
        t1 = time.time()
        
        print(res)
        print(f"[Took {t1-t0:.2f}s]")

if __name__ == "__main__":
    print("Downloading and testing michelinolinolino/gemma4-4b-tikz...")
    test_llama_cpp("michelinolinolino/gemma4-4b-tikz", "model.gguf")
    
    print("\nTesting our base model mlx-community/gemma-4-e4b-it-6bit...")
    test_mlx("mlx-community/gemma-4-e4b-it-6bit")
