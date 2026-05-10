import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Any

def check_render_sanity(pdf_path: str | Path) -> Dict[str, Any]:
    """Check the rendered PDF artifact for basic visual sanity.
    
    Verifies that:
    1. The PDF exists and can be opened.
    2. The rendered image is at least 32x32 pixels.
    3. The image contains a minimum fraction (0.001) of non-background (non-white/transparent) pixels.
    """
    path = Path(pdf_path)
    result = {
        "render_exists": False,
        "width": 0,
        "height": 0,
        "non_background_fraction": 0.0,
        "sanity_passed": False,
        "error": None
    }
    
    if not path.exists():
        result["error"] = "File not found"
        return result
        
    try:
        doc = fitz.open(path)
        if len(doc) == 0:
            result["error"] = "Empty PDF document"
            return result
            
        page = doc[0]
        # Render at default 72 DPI (matrix=fitz.Matrix(1, 1)) is usually enough for sanity check
        pix = page.get_pixmap(alpha=True)
        
        result["render_exists"] = True
        result["width"] = pix.width
        result["height"] = pix.height
        
        if pix.width < 32 or pix.height < 32:
            result["error"] = f"Dimensions too small: {pix.width}x{pix.height}"
            return result
            
        # Count non-white / non-transparent pixels
        # For an RGBA pixmap, each pixel is 4 bytes.
        samples = pix.samples
        total_pixels = pix.width * pix.height
        non_bg_pixels = 0
        
        # We can iterate through the bytes.
        # Pixmap samples is a bytes object of length (width * height * 4) for RGBA
        bytes_per_pixel = pix.n
        
        for i in range(0, len(samples), bytes_per_pixel):
            if bytes_per_pixel >= 3:
                r = samples[i]
                g = samples[i+1]
                b = samples[i+2]
                a = samples[i+3] if bytes_per_pixel == 4 else 255
                
                # If it's transparent, it's background
                if a == 0:
                    continue
                # If it's pure white, it's background
                if r == 255 and g == 255 and b == 255:
                    continue
                non_bg_pixels += 1
            else:
                # grayscale
                val = samples[i]
                a = samples[i+1] if bytes_per_pixel == 2 else 255
                if a == 0 or val == 255:
                    continue
                non_bg_pixels += 1
                
        fraction = non_bg_pixels / float(total_pixels)
        result["non_background_fraction"] = fraction
        
        if fraction >= 0.001:
            result["sanity_passed"] = True
        else:
            result["error"] = f"Non-background fraction too low: {fraction:.5f}"
            
        return result
    except Exception as e:
        result["error"] = str(e)
        return result
