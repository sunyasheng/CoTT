#!/usr/bin/env python3
"""
Overlay image bounding boxes (from parse_pdf_layout.py) onto an image.

Input:
  - --layout: JSON from parse_pdf_layout.py
  - --page: page index (1-based). Defaults to 1.
  - --image: target image path to draw on (PNG/JPG). If omitted, derive from PDF path
            by replacing .pdf with .png in the same directory.
  - --output: output path for the overlaid image. If omitted, appends _boxes to image name.
  - --no-fill: draw only outlines (no translucent fill)
  - --min-area: skip boxes smaller than this ratio of the page (default: 0.0001)
  - --max-area: skip boxes larger than this ratio of the page (default: 0.98)

Coordinates:
  - Uses bbox_normalized (origin top-left, [0,1]) if available; otherwise falls back to bbox_points
    combined with page_size_points to convert to pixel positions.

Dependencies: Pillow (PIL). Install via: pip install pillow
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

try:
    from PIL import Image, ImageDraw
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required. Install with `pip install pillow`.\n"
        f"Import error: {exc}"
    )


def _derive_image_path_from_pdf(pdf_path: str) -> str:
    base, _ = os.path.splitext(pdf_path)
    return base + ".png"


def _load_layout(layout_path: str) -> Dict[str, Any]:
    with open(layout_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_pixel_bbox(img_size: Tuple[int, int], box: Dict[str, float], page: Dict[str, Any]) -> Tuple[int, int, int, int]:
    width, height = img_size
    if "bbox_normalized" in box:
        n = box["bbox_normalized"]
        left = int(round(n["left"] * width))
        top = int(round(n["top"] * height))
        right = int(round(n["right"] * width))
        bottom = int(round(n["bottom"] * height))
        return left, top, right, bottom

    # Fallback: use points with page size
    pts = box.get("bbox_points")
    page_size = page.get("page_size_points", {})
    pw = float(page_size.get("width", 1))
    ph = float(page_size.get("height", 1))
    if not pts or pw <= 0 or ph <= 0:
        return 0, 0, 0, 0

    # Convert bottom-left origin points to top-left origin pixels
    x0 = float(pts["x0"]) / pw * width
    x1 = float(pts["x1"]) / pw * width
    # y from bottom -> from top
    y1_from_bottom = float(pts["y1"]) / ph * height
    y0_from_bottom = float(pts["y0"]) / ph * height
    top = int(round(height - y1_from_bottom))
    bottom = int(round(height - y0_from_bottom))
    left = int(round(x0))
    right = int(round(x1))
    return left, top, right, bottom


def _area_ratio(box: Tuple[int, int, int, int], img_size: Tuple[int, int]) -> float:
    l, t, r, b = box
    w = max(0, r - l)
    h = max(0, b - t)
    iw, ih = img_size
    if iw <= 0 or ih <= 0:
        return 0.0
    return (w * h) / float(iw * ih)


def draw_boxes(layout_path: str, page_index: int, image_path: str | None, output_path: str | None, no_fill: bool, min_area: float, max_area: float) -> str:
    data = _load_layout(layout_path)

    # Locate page
    pages: List[Dict[str, Any]] = data.get("pages", [])
    if not pages:
        raise ValueError("No pages found in layout JSON")
    if page_index < 1 or page_index > len(pages):
        raise ValueError(f"page_index {page_index} out of range (1..{len(pages)})")
    page = pages[page_index - 1]

    # Derive image path if not provided
    if image_path is None:
        pdf_path = data.get("pdf_path")
        if not pdf_path:
            raise ValueError("pdf_path missing in layout JSON; please pass --image explicitly")
        image_path = _derive_image_path_from_pdf(pdf_path)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    boxes: List[Dict[str, Any]] = page.get("images", [])
    W, H = img.size

    # Draw boxes with optional translucent fill and solid border; skip too-small or too-large boxes
    palette = [
        (255, 0, 0, 255),
        (0, 200, 0, 255),
        (0, 120, 255, 255),
        (255, 140, 0, 255),
        (180, 0, 255, 255),
    ]
    for idx, box in enumerate(boxes):
        l, t, r, b = _to_pixel_bbox((W, H), box, page)
        if r <= l or b <= t:
            continue
        ar = _area_ratio((l, t, r, b), (W, H))
        if ar < min_area or ar > max_area:
            continue
        color = palette[idx % len(palette)]
        fill = None if no_fill else (color[0], color[1], color[2], 40)
        draw.rectangle([l, t, r, b], outline=color, width=3, fill=fill)

    # Save
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_boxes{ext}"
    img.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay parsed PDF image boxes onto an image")
    parser.add_argument("--layout", required=True, help="Layout JSON from parse_pdf_layout.py")
    parser.add_argument("--page", type=int, default=1, help="1-based page index")
    parser.add_argument("--image", default=None, help="Image path to draw on (PNG/JPG)")
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument("--no-fill", action="store_true", help="Draw outlines only, no translucent fill")
    parser.add_argument("--min-area", type=float, default=0.0001, help="Skip boxes smaller than this page area ratio")
    parser.add_argument("--max-area", type=float, default=0.98, help="Skip boxes larger than this page area ratio")
    args = parser.parse_args()

    out = draw_boxes(args.layout, args.page, args.image, args.output, args.no_fill, args.min_area, args.max_area)
    print(out)


if __name__ == "__main__":
    main()
