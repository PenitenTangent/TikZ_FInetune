from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
from pathlib import Path

import yaml
from PIL import Image, ImageColor, ImageDraw, ImageOps


class RasterizationError(RuntimeError):
    """Raised when no supported PDF rasterization tool is available."""


RASTER_TIMEOUT_SECONDS = 20
DEFAULT_RENDER_CONFIG_PATH = Path("configs/render_config.yaml")


@dataclass(slots=True)
class RasterizationConfig:
    dpi: int
    page_width_pt: int
    page_height_pt: int
    anti_aliasing: bool
    background: str
    output_format: str
    color_depth: int


@dataclass(slots=True)
class EncoderConfig:
    normalize_l2: bool
    resize_before_encode: tuple[int, int]


@dataclass(slots=True)
class RenderConfig:
    path: Path
    rasterization: RasterizationConfig
    encoder: EncoderConfig


def load_render_config(root_dir: str | Path, config_path: str | Path | None = None) -> RenderConfig:
    root = Path(root_dir).expanduser().resolve()
    path = root / DEFAULT_RENDER_CONFIG_PATH if config_path is None else Path(config_path).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.exists():
        raise RuntimeError(f"Render config is required for evaluation but was not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rasterization = data.get("rasterization") or {}
    encoder = data.get("encoder") or {}
    resize = encoder.get("resize_before_encode", [224, 224])
    return RenderConfig(
        path=path,
        rasterization=RasterizationConfig(
            dpi=int(rasterization.get("dpi", 150)),
            page_width_pt=int(rasterization.get("page_width_pt", 595)),
            page_height_pt=int(rasterization.get("page_height_pt", 842)),
            anti_aliasing=bool(rasterization.get("anti_aliasing", False)),
            background=str(rasterization.get("background", "white")),
            output_format=str(rasterization.get("output_format", "png")).lower(),
            color_depth=int(rasterization.get("color_depth", 8)),
        ),
        encoder=EncoderConfig(
            normalize_l2=bool(encoder.get("normalize_l2", True)),
            resize_before_encode=(int(resize[0]), int(resize[1])),
        ),
    )


def _pick_rasterizer() -> str | None:
    for binary in ("pdftoppm", "mutool", "magick"):
        if shutil.which(binary):
            return binary
    return None


def rasterize_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    render_config: RenderConfig | None = None,
) -> Path:
    pdf = Path(pdf_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_png = target_dir / f"{pdf.stem}.png"

    if render_config is not None:
        return _rasterize_with_pdfium(pdf, output_png, render_config=render_config)

    rasterizer = _pick_rasterizer()
    if rasterizer is None:
        return _rasterize_with_pdfium(pdf, output_png)

    if rasterizer == "pdftoppm":
        base = output_png.with_suffix("")
        subprocess.run(
            ["pdftoppm", "-singlefile", "-png", str(pdf), str(base)],
            check=True,
            capture_output=True,
            text=True,
            timeout=RASTER_TIMEOUT_SECONDS,
        )
        return output_png

    if rasterizer == "mutool":
        subprocess.run(
            ["mutool", "draw", "-o", str(output_png), "-F", "png", str(pdf), "1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=RASTER_TIMEOUT_SECONDS,
        )
        return output_png

    subprocess.run(
        ["magick", "-density", "200", str(pdf), str(output_png)],
        check=True,
        capture_output=True,
        text=True,
        timeout=RASTER_TIMEOUT_SECONDS,
    )
    return output_png


def _rasterize_with_pdfium(
    pdf_path: Path,
    output_png: Path,
    render_config: RenderConfig | None = None,
) -> Path:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RasterizationError(
            "No PDF rasterizer found. Install `pdftoppm`, `mutool`, `magick`, or the Python package `pypdfium2`."
        ) from exc

    document = pdfium.PdfDocument(str(pdf_path))
    page = document[0]
    try:
        if render_config is None:
            bitmap = page.render(scale=2.0).to_pil()
        else:
            raster = render_config.rasterization
            bitmap = page.render(
                scale=raster.dpi / 72.0,
                fill_color=(*ImageColor.getrgb(raster.background), 255),
                rev_byteorder=True,
                no_smoothtext=not raster.anti_aliasing,
                no_smoothimage=not raster.anti_aliasing,
                no_smoothpath=not raster.anti_aliasing,
            ).to_pil()
            bitmap = _apply_fixed_raster_canvas(bitmap.convert("RGB"), raster)
    finally:
        close_page = getattr(page, "close", None)
        if callable(close_page):
            close_page()
        close_document = getattr(document, "close", None)
        if callable(close_document):
            close_document()
    bitmap.save(output_png)
    return output_png


def _apply_fixed_raster_canvas(image: Image.Image, config: RasterizationConfig) -> Image.Image:
    target_width = max(1, int(round(config.page_width_pt * config.dpi / 72.0)))
    target_height = max(1, int(round(config.page_height_pt * config.dpi / 72.0)))
    background = ImageColor.getrgb(config.background)
    if image.width > target_width or image.height > target_height:
        resample = Image.Resampling.BILINEAR if config.anti_aliasing else Image.Resampling.NEAREST
        image = ImageOps.contain(image, (target_width, target_height), method=resample)

    canvas = Image.new("RGB", (target_width, target_height), color=background)
    offset = ((target_width - image.width) // 2, (target_height - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def prepare_image_for_reward_encoder(image_path: str | Path, render_config: RenderConfig) -> Path:
    target = Path(image_path)
    desired_size = render_config.encoder.resize_before_encode
    with Image.open(target) as image_handle:
        image = image_handle.convert("RGB")
        if image.size == desired_size:
            return target
        resample = (
            Image.Resampling.BILINEAR
            if render_config.rasterization.anti_aliasing
            else Image.Resampling.NEAREST
        )
        resized = image.resize(desired_size, resample=resample)

    output_path = target.with_name(f"{target.stem}.encoder.png")
    resized.save(output_path, format="PNG")
    return output_path


def overlay_debug_grid(image_path: str | Path, output_path: str | Path, step_px: int = 32) -> Path:
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for x in range(0, width, step_px):
        color = (220, 220, 220, 110)
        line_width = 2 if x == width // 2 else 1
        draw.line([(x, 0), (x, height)], fill=color, width=line_width)

    for y in range(0, height, step_px):
        color = (220, 220, 220, 110)
        line_width = 2 if y == height // 2 else 1
        draw.line([(0, y), (width, y)], fill=color, width=line_width)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def build_debug_render(pdf_path: str | Path, output_dir: str | Path, step_px: int = 32) -> Path:
    rasterized = rasterize_pdf(pdf_path, output_dir)
    debug_path = Path(output_dir) / f"{Path(pdf_path).stem}.grid.png"
    return overlay_debug_grid(rasterized, debug_path, step_px=step_px)
