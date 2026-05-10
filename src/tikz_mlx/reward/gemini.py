import os
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

class GeminiVisiionReward:
    """
    Experimental reward backend using Gemini 1.5/2.0 Flash Lite API.
    Replaces local VRAM-heavy encoders with a cloud-based vision judge.
    """
    def __init__(self, model_name: str = "gemini-3.1-flash-lite-preview", api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing Gemini API Key. Please set GOOGLE_GENERATIVE_AI_API_KEY in your .env file "
                "or pass it to the constructor."
            )
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
        self.prompt = (
            "You are a TikZ diagram expert. Compare the two provided images.\n"
            "Image 1: Reference render (Ground Truth).\n"
            "Image 2: Candidate render (Model Output).\n\n"
            "Rate the visual and structural similarity from 0.0 to 1.0.\n"
            "1.0: Identical or functionally perfect match.\n"
            "0.8: High similarity, maybe minor coordinate shifts or label sizing differences.\n"
            "0.5: Correct general structure but missing major components or wrong colors.\n"
            "0.0: Completely wrong or empty.\n\n"
            "Only output the number (e.g. 0.85)."
        )

    def score_images(self, reference_path: Path, candidate_path: Path) -> float:
        """
        Calculates similarity score by sending images to Gemini.
        """
        try:
            img1 = Image.open(reference_path)
            img2 = Image.open(candidate_path)
            
            response = self.model.generate_content(
                [self.prompt, img1, img2]
            )
            
            # Extract number from response
            text = response.text.strip()
            # Simple numeric extraction
            import re
            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
            return 0.0
        except Exception as e:
            print(f"Gemini Reward Error: {e}")
            return 0.0

# Singleton-like accessor for the evaluator
_GEMINI_REWARD = None

def get_gemini_reward() -> GeminiVisiionReward:
    global _GEMINI_REWARD
    if _GEMINI_REWARD is None:
        _GEMINI_REWARD = GeminiVisiionReward()
    return _GEMINI_REWARD
