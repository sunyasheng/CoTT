#!/usr/bin/env python3
import os
import re
import json
import argparse
import base64
from pathlib import Path


def load_env() -> str | None:
    try:
        from dotenv import load_dotenv, find_dotenv  # optional
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path)
            return env_path
    except Exception:
        return None
    return None


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Markdown not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_figure_mentions(text: str):
    lines = text.splitlines()
    candidates = []
    pat = re.compile(r"\b(Figure|Fig\.)\s*(\d{1,3})\b", re.IGNORECASE)
    for i, _line in enumerate(lines):
        if pat.search(_line):
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            snippet = "\n".join(lines[start:end]).strip()
            nums = [m.group(2) for m in pat.finditer(snippet)]
            candidates.append({
                "line": i + 1,
                "snippet": snippet,
                "fig_nums": list(dict.fromkeys(nums)),
            })
    return candidates


def extract_context_for_figure(md_text: str, figure_num: str, window: int = 6) -> str:
    lines = md_text.splitlines()
    pat = re.compile(rf"\b(Figure|Fig\.)\s*{re.escape(figure_num)}\b", re.IGNORECASE)
    first_idx = -1
    for i, line in enumerate(lines):
        if pat.search(line):
            first_idx = i
            break
    if first_idx == -1:
        return ""
    start = max(0, first_idx - window)
    end = min(len(lines), first_idx + window + 1)
    snippet = lines[start:end]
    # Also gather nearby markdown image lines
    img_pat = re.compile(r"!\[[^\]]*\]\([^\)]+\)")
    extra = []
    for j in range(max(0, first_idx - window * 2), min(len(lines), first_idx + window * 2)):
        if img_pat.search(lines[j]):
            extra.append(lines[j])
    context = "\n".join(snippet + (["\nImages:"] + extra if extra else []))
    return context.strip()


def find_figure_images(md_path: Path, md_text: str, figure_num: str) -> list[Path]:
    lines = md_text.splitlines()
    pat = re.compile(rf"\b(Figure|Fig\.)\s*{re.escape(figure_num)}\b", re.IGNORECASE)
    img_pat = re.compile(r"!\[[^\]]*\]\(([^\)]+)\)")
    first_idx = -1
    for i, line in enumerate(lines):
        if pat.search(line):
            first_idx = i
            break
    if first_idx == -1:
        return []
    images: list[Path] = []
    base = md_path.parent
    for j in range(max(0, first_idx - 15), min(len(lines), first_idx + 15)):
        m = img_pat.search(lines[j])
        if m:
            raw = m.group(1).strip()
            img_rel = raw.split()[0]
            p = (base / img_rel).resolve()
            if p.exists() and p.is_file():
                images.append(p)
    return images[:3]


def encode_image_to_data_url(image_path: Path) -> str:
    with image_path.open('rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    media = f"image/{image_path.suffix.lstrip('.').lower()}" if image_path.suffix else "image/png"
    return f"data:{media};base64,{b64}"


def call_qwen_vl(model: str, messages: list, timeout: int = 120) -> dict:
    # Qwen OpenAI-compatible endpoint
    from openai import OpenAI
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    base_url = os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    else:
        client = OpenAI(api_key=api_key, timeout=timeout)

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)


def build_caption_prompt(fig_num: str, context: str, max_captions: int) -> str:
    guide = (
        "You are given the actual figure image plus nearby markdown context.\n"
        "Produce 2-3 diverse drawing-oriented captions that would let a designer or a text-to-image model recreate the figure layout.\n"
        "Each caption must be step-by-step, covering: components, spatial arrangement (left/center/right, grid/columns/lanes), arrows/edges (directions and labels), grouping/containers, legend, title, key labels, and stylistic hints (shapes, colors, line styles).\n"
        "Prefer concise but complete instructions. Avoid results plots; focus on system/flow structure.\n"
        f"Return strict JSON with keys: figure (string), captions (array, up to {max_captions})."
    )
    ctx = f"Figure: {fig_num}\nContext from paper markdown (may include nearby text or image markdown):\n---\n{context}\n---"
    return guide + "\n\n" + ctx


def main():
    parser = argparse.ArgumentParser(description="Generate figure captions using Qwen2.5-VL (OpenAI-compatible API)")
    parser.add_argument("--md", required=True, help="Path to the paper markdown")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Qwen VL model name")
    parser.add_argument("--max_figures", type=int, default=3, help="Max figures to process")
    parser.add_argument("--max_captions", type=int, default=3, help="Max captions per figure")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout seconds")
    args = parser.parse_args()

    env_path = load_env()
    if env_path:
        print(f"Loaded .env: {env_path}")

    md_path = Path(args.md)
    md_text = read_text(md_path)

    # Simple selection: pick up to max_figures mentioned in order
    mentions = extract_figure_mentions(md_text)
    # deduplicate by figure number order of appearance
    ordered_figs = []
    for c in mentions:
        for n in c["fig_nums"]:
            if n not in ordered_figs:
                ordered_figs.append(n)
    figures = ordered_figs[: args.max_figures]
    if not figures:
        print(json.dumps({"figures": [], "note": "No figure mentions found."}, ensure_ascii=False))
        return

    outputs = []
    for fig in figures:
        context = extract_context_for_figure(md_text, str(fig))
        image_paths = find_figure_images(md_path, md_text, str(fig))
        # Build OpenAI-style messages with images as data URLs
        user_content = [{"type": "text", "text": build_caption_prompt(str(fig), context, args.max_captions)}]
        for p in image_paths:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": encode_image_to_data_url(p)}
            })
        messages = [
            {"role": "system", "content": "You are a precise assistant that outputs strict JSON only."},
            {"role": "user", "content": user_content},
        ]

        obj = call_qwen_vl(args.model, messages, timeout=args.timeout)
        outputs.append({
            "figure": str(fig),
            "captions": obj.get("captions", [])[: args.max_captions],
            "images": [str(p) for p in image_paths],
            "context_preview": context[:400],
        })

    print(json.dumps({"figures": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    main()


