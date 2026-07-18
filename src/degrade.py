"""Robustness degradations.

The dataset images are pristine 200-DPI renders, so the robustness axis needs
synthesized degradation: rotation, blur, sensor noise, downscaling (low-DPI
scan), and JPEG compression. Each named perturbation maps a clean PIL image to a
degraded one; the benchmark compares clean vs degraded extraction accuracy.

Pure PIL + numpy (no OpenCV dependency).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

WHITE = (255, 255, 255)


def _rotate(deg: float) -> Callable[[Image.Image], Image.Image]:
    def f(img: Image.Image) -> Image.Image:
        return img.convert("RGB").rotate(deg, expand=True, fillcolor=WHITE, resample=Image.BICUBIC)
    return f


def _blur(radius: float) -> Callable[[Image.Image], Image.Image]:
    return lambda img: img.convert("RGB").filter(ImageFilter.GaussianBlur(radius))


def _noise(sigma: float) -> Callable[[Image.Image], Image.Image]:
    def f(img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB")).astype(np.int16)
        rng = np.random.default_rng(0)  # fixed so degradation is reproducible
        arr = arr + rng.normal(0, sigma, arr.shape).astype(np.int16)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    return f


def _downscale(target_dpi: int, src_dpi: int = 200) -> Callable[[Image.Image], Image.Image]:
    """Simulate a lower-DPI scan: shrink then blur slightly (no upscale back)."""
    scale = target_dpi / src_dpi
    def f(img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        return small.filter(ImageFilter.GaussianBlur(0.4))
    return f


def _jpeg(quality: int) -> Callable[[Image.Image], Image.Image]:
    def f(img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return f


def _compose(*fns: Callable[[Image.Image], Image.Image]) -> Callable[[Image.Image], Image.Image]:
    def f(img: Image.Image) -> Image.Image:
        for fn in fns:
            img = fn(img)
        return img
    return f


# Named perturbations used as the robustness axis in the benchmark.
PERTURBATIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "clean": lambda img: img.convert("RGB"),
    "rotate5": _rotate(5),
    "rotate15": _rotate(15),
    "blur": _blur(1.6),
    "noise": _noise(18),
    "downscale150": _downscale(150),
    "downscale100": _downscale(100),
    "jpeg40": _jpeg(40),
    "scan_combo": _compose(_rotate(3), _downscale(150), _noise(10), _jpeg(55)),  # realistic scan
}


def apply(img: Image.Image, kind: str) -> Image.Image:
    if kind not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {kind!r}; options: {list(PERTURBATIONS)}")
    return PERTURBATIONS[kind](img)


def degrade_file(src_png: str | Path, kind: str, dst_png: str | Path) -> Path:
    dst_png = Path(dst_png)
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    out = apply(Image.open(src_png), kind)
    out.save(dst_png)
    return dst_png


if __name__ == "__main__":
    from src import dataset as ds

    d = ds.list_docs("bank_statement")[0]
    outdir = Path("data/perturbed/_demo")
    print(f"source: {d.image_path.name}")
    for kind in PERTURBATIONS:
        p = degrade_file(d.image_path, kind, outdir / f"{d.stem}__{kind}.png")
        w, h = Image.open(p).size
        print(f"  {kind:14s} -> {p.name}  ({w}x{h})")
