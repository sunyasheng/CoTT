#!/usr/bin/env python3
"""
Parse a PDF and extract element positions relative to the visible page area (CropBox).

Primary goal: report image placements (bounding boxes) per page, with coordinates
relative to the page CropBox. Coordinates are provided in:
- pdf_points: absolute PDF points (1/72 in) in the PDF native coordinate system
              origin at bottom-left of the visible page (CropBox)
- normalized: [0,1] relative coordinates with origin at top-left of the visible page
              (i.e., normalized_top = y_from_top / page_height)

Requires: PyMuPDF (fitz). Install via: pip install pymupdf

Usage:
  python parse_pdf_layout.py --pdf /path/to/file.pdf --output layout.json

Notes:
  - We use page.rect (CropBox) as the visible area. If CropBox is undefined,
    fitz uses the effective page rectangle (falls back to MediaBox internally).
  - For images, we rely on page.get_images + page.get_image_bbox(xref) to obtain
    placements. If your PyMuPDF version supports page.get_image_info(full=True),
    this script automatically prefers it.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyMuPDF (fitz) is required. Install with `pip install pymupdf`.\n"
        f"Import error: {exc}"
    )


def _normalize_bbox_to_top_left(bbox: fitz.Rect, page_rect: fitz.Rect) -> Dict[str, float]:
    width = float(page_rect.width)
    height = float(page_rect.height)
    if width <= 0 or height <= 0:
        return {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}

    # PDF origin is bottom-left. For top-left normalized coords:
    # top = (page_top - bbox_top) but since y increases upwards,
    # y_from_top = page_rect.y1 - bbox.y1
    left = (float(bbox.x0) - float(page_rect.x0)) / width
    right = (float(bbox.x1) - float(page_rect.x0)) / width
    top = (float(page_rect.y1) - float(bbox.y1)) / height
    bottom = (float(page_rect.y1) - float(bbox.y0)) / height

    # Clamp to [0,1] to avoid tiny numeric spillover
    def clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    return {
        "left": clamp(left),
        "top": clamp(top),
        "right": clamp(right),
        "bottom": clamp(bottom),
    }


def _rect_to_points_from_bottom_left(bbox: fitz.Rect, page_rect: fitz.Rect) -> Dict[str, float]:
    # Translate to page cropbox bottom-left origin
    return {
        "x0": float(bbox.x0 - page_rect.x0),
        "y0": float(bbox.y0 - page_rect.y0),
        "x1": float(bbox.x1 - page_rect.x0),
        "y1": float(bbox.y1 - page_rect.y0),
    }


def _area_ratio_points(pts: Dict[str, float], page_size: Tuple[float, float]) -> float:
    pw, ph = page_size
    if pw <= 0 or ph <= 0:
        return 0.0
    w = max(0.0, float(pts["x1"]) - float(pts["x0"]))
    h = max(0.0, float(pts["y1"]) - float(pts["y0"]))
    return (w * h) / (pw * ph)


def _dedup_by_normalized(boxes: List[Dict[str, Any]], precision: int = 3) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for b in boxes:
        n = b.get("bbox_normalized") or {}
        key = (
            round(float(n.get("left", 0)), precision),
            round(float(n.get("top", 0)), precision),
            round(float(n.get("right", 0)), precision),
            round(float(n.get("bottom", 0)), precision),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(b)
    return unique


def parse_pdf(
    pdf_path: str,
    min_area: float = 0.0,
    max_area: float = 1.0,
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    pages: List[Dict[str, Any]] = []

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        page_rect = page.rect  # this reflects CropBox / visible area

        images_info: List[Dict[str, Any]] = []

        # Prefer newer PyMuPDF API if present
        img_entries = []
        if hasattr(page, "get_image_info"):
            try:
                img_entries = page.get_image_info(xrefs=True)  # type: ignore[arg-type]
            except TypeError:
                img_entries = page.get_image_info()  # type: ignore[call-arg]

        if img_entries:
            for info in img_entries:
                # info may contain: xref, bbox, transform, width, height, etc.
                xref = info.get("xref")
                bbox = info.get("bbox")
                if bbox is None:
                    # Convert from tuple to Rect if needed
                    try:
                        bbox = fitz.Rect(info["x0"], info["y0"], info["x1"], info["y1"])  # type: ignore[index]
                    except Exception:
                        continue
                rect = fitz.Rect(bbox)
                pts = _rect_to_points_from_bottom_left(rect, page_rect)
                images_info.append(
                    {
                        "bbox_points": pts,
                        "size_points": {
                            "width": round(pts["x1"] - pts["x0"], 3),
                            "height": round(pts["y1"] - pts["y0"], 3),
                        },
                        "bbox_normalized": _normalize_bbox_to_top_left(rect, page_rect),
                    }
                )
        else:
            # Fallback for older PyMuPDF: iterate xrefs, then get_image_bbox
            seen_xrefs = set()
            for item in page.get_images(full=True):
                # item structure per docs: (xref, smask, width, height, bpc, colorspace, alt, name)
                xref = int(item[0])
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    rect = page.get_image_bbox(xref)
                except Exception:
                    continue
                pts = _rect_to_points_from_bottom_left(rect, page_rect)
                images_info.append(
                    {
                        "bbox_points": pts,
                        "size_points": {
                            "width": round(pts["x1"] - pts["x0"], 3),
                            "height": round(pts["y1"] - pts["y0"], 3),
                        },
                        "bbox_normalized": _normalize_bbox_to_top_left(rect, page_rect),
                    }
                )

        # Filtering by area and deduplication
        pw, ph = float(page_rect.width), float(page_rect.height)
        filtered: List[Dict[str, Any]] = []
        for b in images_info:
            ar = _area_ratio_points(b["bbox_points"], (pw, ph))
            if ar < min_area or ar > max_area:
                continue
            b["area_ratio"] = round(ar, 6)
            filtered.append(b)

        # Sort by area descending and apply top_n if provided
        filtered.sort(key=lambda x: x.get("area_ratio", 0.0), reverse=True)
        if top_n is not None and top_n > 0:
            filtered = filtered[:top_n]

        # Deduplicate similar boxes
        filtered = _dedup_by_normalized(filtered, precision=3)

        pages.append(
            {
                "page_index": page_index + 1,
                "page_size_points": {
                    "width": float(page_rect.width),
                    "height": float(page_rect.height),
                },
                "images": filtered,
            }
        )

    doc.close()
    return {"pdf_path": pdf_path, "num_pages": len(pages), "pages": pages}


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PDF and extract element positions relative to CropBox")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--output", default=None, help="Output JSON path (prints to stdout if omitted)")
    parser.add_argument("--min-area", type=float, default=0.0, help="Keep boxes with page area ratio >= this value")
    parser.add_argument("--max-area", type=float, default=1.0, help="Keep boxes with page area ratio <= this value")
    parser.add_argument("--top-n", type=int, default=None, help="Keep only the largest N boxes per page after filtering")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        raise FileNotFoundError(args.pdf)

    report = parse_pdf(args.pdf, min_area=args["min_area"] if hasattr(args, "__getitem__") else args.min_area, max_area=args["max_area"] if hasattr(args, "__getitem__") else args.max_area, top_n=args["top_n"] if hasattr(args, "__getitem__") else args.top_n)
    data = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(data)
    else:
        print(data)


if __name__ == "__main__":
    main()


