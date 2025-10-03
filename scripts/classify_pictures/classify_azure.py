#!/usr/bin/env python3
"""
Image classifier for paper figures referenced in a pandoc-converted markdown file.

Given a paper directory that contains `output_pandoc.md`, this script:
1) Parses image references and their nearby captions/text
2) Applies keyword-based heuristics to classify each image as one of:
   - teaser
   - workflow
   - diagram
   - qualitative
3) Emits a JSON report with labels, confidences, and rationale

Usage:
  python classify_azure.py --paper_dir /path/to/paper_dir \
      --output report.json

Notes:
  - This is a lightweight, dependency-free heuristic classifier.
  - It considers position (earlier figures more likely to be teaser) and keywords.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple


IMAGE_MD_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
# HTML <embed> or <img> with src="..."
IMAGE_HTML_PATTERN = re.compile(r"<\s*(?:embed|img)[^>]*?src=\"(?P<path>[^\"]+)\"[^>]*?>", re.IGNORECASE)
# <figcaption> ... </figcaption>
FIGCAPTION_INLINE_PATTERN = re.compile(r"<\s*figcaption\s*>\s*(?P<cap>.*?)\s*<\s*/\s*figcaption\s*>", re.IGNORECASE)
FIGURE_LABEL_PATTERN = re.compile(r"^\s*(Figure|Fig\.|Fig)\s*\d+\s*[:.\-]\s*(?P<caption>.*)$", re.IGNORECASE)
INLINE_FIGURE_PATTERN = re.compile(r"\b(Figure|Fig\.|Fig)\s*(?P<num>\d+)\b", re.IGNORECASE)


@dataclass
class ImageEntry:
    figure_index: int
    image_path: str
    alt_text: str
    caption: str
    surrounding_text: str


def read_markdown(md_path: str) -> List[str]:
    with open(md_path, "r", encoding="utf-8") as f:
        return f.readlines()


def _ensure_png_for_pdf(rel_path: str, paper_dir: str) -> str:
    """Ensure a PNG exists for a given relative PDF path.

    Returns a relative path to the PNG to use in the report. If conversion
    fails or pdftoppm is unavailable, falls back to replacing extension.
    """
    if not rel_path.lower().endswith(".pdf"):
        return rel_path

    pdf_rel = rel_path
    png_rel = rel_path[:-4] + ".png"

    pdf_abs = os.path.join(paper_dir, pdf_rel)
    png_abs = os.path.join(paper_dir, png_rel)

    # If desired PNG already exists, just use it
    if os.path.isfile(png_abs):
        return png_rel

    # If PDF doesn't exist, nothing we can do; return the .png path as best-effort
    if not os.path.isfile(pdf_abs):
        return png_rel

    # Try conversion if pdftoppm is available
    if shutil.which("pdftoppm") is None:
        return png_rel

    # Run pdftoppm to convert; it will output base-1.png, base-2.png, ...
    out_base_abs = os.path.splitext(pdf_abs)[0]
    try:
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r",
                "300",
                "-cropbox",
                pdf_abs,
                out_base_abs,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return png_rel

    # Decide which output to reference
    single_page = os.path.isfile(f"{out_base_abs}-1.png") and not any(
        os.path.isfile(f"{out_base_abs}-{n}.png") for n in range(2, 6)
    )

    if single_page:
        # For single-page, prefer name.png; rename if needed
        first_png_abs = f"{out_base_abs}-1.png"
        if not os.path.isfile(png_abs) and os.path.isfile(first_png_abs):
            try:
                os.replace(first_png_abs, png_abs)
            except Exception:
                # If rename fails, fall back to referencing -1.png
                return os.path.relpath(first_png_abs, paper_dir)
        return png_rel

    # Multi-page: keep -1.png as reference
    first_png_abs = f"{out_base_abs}-1.png"
    if os.path.isfile(first_png_abs):
        return os.path.relpath(first_png_abs, paper_dir)

    # Fallback to simple extension replacement
    return png_rel


def _collect_context(lines: List[str], idx: int, window: int = 3) -> str:
    start = max(0, idx - window)
    end = min(len(lines), idx + window + 1)
    return " ".join(line.strip() for line in lines[start:end])


def _strip_html_tags(text: str) -> str:
    # Very light HTML tag stripper for captions
    return re.sub(r"<[^>]+>", " ", text).strip()


def _find_nearby_figcaption(lines: List[str], i: int, search_window: int = 8) -> Optional[str]:
    start = max(0, i - search_window)
    end = min(len(lines), i + search_window + 1)
    window_text = " ".join(lines[start:end])
    m = FIGCAPTION_INLINE_PATTERN.search(window_text)
    if m:
        return _strip_html_tags(m.group("cap"))
    return None


def parse_images_with_captions(lines: List[str], paper_dir: str) -> List[ImageEntry]:
    entries: List[ImageEntry] = []
    figure_counter = 0

    for i, line in enumerate(lines):
        alt_text = ""
        image_path: Optional[str] = None

        img_match = IMAGE_MD_PATTERN.search(line)
        html_match = IMAGE_HTML_PATTERN.search(line)

        if img_match:
            alt_text = img_match.group("alt").strip()
            image_path = img_match.group("path").strip()
        elif html_match:
            image_path = html_match.group("path").strip()
        else:
            continue

        # Normalize and ensure PNG if the source was a PDF
        if image_path.lower().endswith(".pdf"):
            image_path = _ensure_png_for_pdf(image_path, paper_dir)

        # Attempt to find a caption: prefer explicit figcaption or "Figure X:" lines near the image
        caption_candidates: List[Tuple[int, str]] = []
        # Check explicit figcaption in a nearby window (works for <figure> blocks)
        figcap = _find_nearby_figcaption(lines, i, search_window=8)
        if figcap:
            caption_candidates.append((i, figcap))

        # Also look for textual Figure lines near the image position
        for j in range(max(0, i - 3), min(len(lines), i + 9)):
            m = FIGURE_LABEL_PATTERN.match(lines[j].strip())
            if m:
                caption_candidates.append((j, m.group("caption").strip()))

        if caption_candidates:
            # Choose the closest caption line (or the figcaption we injected at i)
            caption_candidates.sort(key=lambda t: abs(t[0] - i))
            caption = caption_candidates[0][1]
        else:
            # Fallback to alt text or nearby paragraph
            nearby = _collect_context(lines, i, window=3)
            caption = alt_text if alt_text else nearby

        figure_counter += 1
        entries.append(
            ImageEntry(
                figure_index=figure_counter,
                image_path=image_path,
                alt_text=alt_text,
                caption=_strip_html_tags(caption),
                surrounding_text=_strip_html_tags(_collect_context(lines, i, window=6)),
            )
        )

    return entries


def _score(text: str, keywords: List[str]) -> int:
    text_lower = text.lower()
    score = 0
    for kw in keywords:
        if kw in text_lower:
            score += 1
    return score


def classify_entry(entry: ImageEntry, total_count: int) -> Dict[str, Any]:
    text = " ".join([
        entry.alt_text,
        entry.caption,
        entry.surrounding_text,
        os.path.basename(entry.image_path),
    ]).lower()

    # Heuristic keyword groups
    teaser_keywords = [
        "teaser", "at a glance", "glance", "overview figure", "front page",
        "cover", "spotlight",
    ]

    workflow_keywords = [
        "pipeline", "workflow", "architecture", "framework", "overview",
        "model structure", "design", "module", "stage", "encoder", "decoder",
        "block diagram", "data flow", "training scheme", "inference pipeline",
    ]

    diagram_keywords = [
        "diagram", "schematic", "flowchart", "chart", "graph structure",
        "layout", "state machine", "uml", "illustration",
    ]

    qualitative_keywords = [
        "qualitative", "visual results", "visual comparison", "examples",
        "case study", "comparison", "ours vs", "user study", "sample results",
        "failure case", "success case", "more results", "ablation visual",
    ]

    # Scoring
    scores = {
        "teaser": _score(text, teaser_keywords),
        "workflow": _score(text, workflow_keywords),
        "diagram": _score(text, diagram_keywords),
        "qualitative": _score(text, qualitative_keywords),
    }

    # Positional prior: first figure is often a teaser or workflow/overview
    if entry.figure_index == 1:
        scores["teaser"] += 1
        scores["workflow"] += 1

    # Filename cues
    basename = os.path.basename(entry.image_path).lower()
    if any(kw in basename for kw in ["teaser", "front", "overview"]):
        scores["teaser"] += 2
        scores["workflow"] += 1
    if any(kw in basename for kw in ["pipeline", "arch", "framework", "structure", "model"]):
        scores["workflow"] += 2
    if "diagram" in basename:
        scores["diagram"] += 2
    if any(kw in basename for kw in ["qual", "qualitative", "results", "example", "comparison"]):
        scores["qualitative"] += 2

    # Derive label with deterministic tie-breaking
    label_order = ["teaser", "workflow", "diagram", "qualitative"]
    max_score = max(scores.values()) if scores else 0
    candidate_labels = [k for k, v in scores.items() if v == max_score]
    for label in label_order:
        if label in candidate_labels:
            chosen = label
            break
    else:
        chosen = "qualitative"  # conservative fallback

    # Confidence: normalized simple heuristic
    total_possible = 8  # rough scale; not a probability
    confidence = min(1.0, max_score / total_possible if total_possible else 0.0)

    rationale = {
        "scores": scores,
        "position_bias": entry.figure_index,
        "basename": basename,
    }

    return {
        "figure_index": entry.figure_index,
        "image_path": entry.image_path,
        "alt_text": entry.alt_text,
        "caption": entry.caption,
        "label": chosen,
        "confidence": round(confidence, 3),
        "rationale": rationale,
    }


def classify_markdown(md_path: str) -> Dict[str, Any]:
    lines = read_markdown(md_path)
    paper_dir = os.path.dirname(md_path)
    entries = parse_images_with_captions(lines, paper_dir)
    total = len(entries)
    results = [classify_entry(e, total) for e in entries]
    return {
        "markdown_path": md_path,
        "num_images": total,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify images in output_pandoc.md")
    parser.add_argument("--paper_dir", help="Directory containing output_pandoc.md")
    parser.add_argument("--root_dir", help="Process all subdirectories containing output_pandoc.md under this root")
    parser.add_argument("--output", default=None, help="Path to write JSON report; prints to stdout if omitted (single mode)")
    args = parser.parse_args()

    if not args.paper_dir and not args.root_dir:
        raise SystemExit("Please specify --paper_dir or --root_dir")

    if args.paper_dir:
        md_path = os.path.join(args.paper_dir, "output_pandoc.md")
        if not os.path.isfile(md_path):
            raise FileNotFoundError(f"Markdown file not found: {md_path}")
        report = classify_markdown(md_path)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    # Batch mode
    root = args.root_dir
    processed: Dict[str, str] = {}
    for entry in sorted(os.listdir(root)):
        paper_dir = os.path.join(root, entry)
        if not os.path.isdir(paper_dir):
            continue
        md_path = os.path.join(paper_dir, "output_pandoc.md")
        if not os.path.isfile(md_path):
            continue
        try:
            report = classify_markdown(md_path)
            out_path = os.path.join(paper_dir, "classify_report.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            processed[paper_dir] = "ok"
        except Exception as e:
            processed[paper_dir] = f"error: {e}"

    print(json.dumps({"root_dir": root, "processed": processed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


