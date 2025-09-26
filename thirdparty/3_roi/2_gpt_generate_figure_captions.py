#!/usr/bin/env python3
import os
import re
import json
import argparse
import subprocess
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


def run_select_figures(md_path: Path, model: str) -> dict:
    """Call select_figures.py and return its JSON result."""
    script = Path(__file__).parent / "gpt_select_figures.py"
    if not script.exists():
        raise FileNotFoundError(f"select_figures.py not found at {script}")
    cmd = [
        "python3",
        str(script),
        "--md",
        str(md_path),
        "--model",
        model,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"select_figures failed: {res.stderr}")
    try:
        return json.loads(res.stdout.strip())
    except Exception as e:
        raise RuntimeError(f"Invalid JSON from select_figures: {e}; raw={res.stdout[:300]}")


def extract_context_for_figure(md_text: str, figure_num: str, window: int = 6) -> str:
    """Extract lines around the first mention of Figure <num> as context, plus image/caption lines if present."""
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


def encode_image_to_base64(image_path: Path) -> str:
    with image_path.open('rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def call_openai(model: str, system_prompt: str, user_prompt: str, images: list[Path] | None = None, timeout: int = 90) -> dict:
    try:
        from openai import OpenAI
        client = OpenAI()
        user_content = [{"type": "text", "text": user_prompt}]
        if images:
            for img in images:
                b64 = encode_image_to_base64(img)
                media = f"image/{img.suffix.lstrip('.').lower()}" if img.suffix else "image/png"
                data_url = f"data:{media};base64,{b64}"
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            timeout=timeout,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")


def build_caption_prompt(fig_num: str, context: str, max_captions: int) -> str:
    cap_hint = (
        f"Provide at least {max_captions} steps." if max_captions and max_captions > 0
        else "Provide as many atomic steps as needed (no hard upper bound)."
    )
    guide = (
        "You are given the actual figure image plus nearby markdown context from a paper.\n"
        "Role: a senior illustrator who must (1) build a rubric, then (2) reverse-engineer drawing instructions.\n\n"
        "1) Rubric:\n"
        "- Derive the rubric from the paper's logic (not from guesswork). Map textual semantics (entities, relations, process steps, constraints) to visual requirements (what must appear, where, how connected, with which labels/symbols).\n"
        "- The rubric should specify necessary conditions for correctness: entities present, spatial relations, arrow directions/labels, grouping/containers, legends/titles/keys, math/symbols, styling fidelity; plus any ordering/causal constraints implied by the paper.\n\n"
        "2) Reverse-Engineering (Drawing Instructions):\n"
        "- Produce an exhaustive, step-by-step set of drawing instructions to recreate the figure. " + cap_hint + "\n"
        "- Cover: components, spatial arrangement (left/center/right, grid/columns/lanes), arrows/edges (directions & labels), grouping/containers, legend/title/key labels, and stylistic hints (shape/color/line style).\n"
        "- Also provide one consolidated 'reconstruction_prompt' suitable for a text-to-image model (concise but complete).\n\n"
        "Return strict JSON with keys: figure (string), rubric (object), captions (array of steps), reconstruction_prompt (string)."
    )
    ctx = f"Figure: {fig_num}\nContext from paper markdown (may include nearby text or image markdown):\n---\n{context}\n---"
    return guide + "\n\n" + ctx


def find_figure_images(md_path: Path, md_text: str, figure_num: str) -> list[Path]:
    """Find image files referenced near the figure mention. Resolve paths relative to md file folder."""
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
            # strip possible title after space
            img_rel = raw.split()[0]
            p = (base / img_rel).resolve()
            if p.exists() and p.is_file():
                images.append(p)
    return images[:3]


def main():
    parser = argparse.ArgumentParser(description="Generate 2-3 draw-focused captions for selected figures.")
    parser.add_argument("--md", required=True, help="Path to the paper markdown")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"), help="OpenAI model")
    parser.add_argument("--timeout", type=int, default=120, help="OpenAI request timeout seconds")
    parser.add_argument("--max_captions", type=int, default=3, help="Max captions per figure (2-3 recommended)")
    args = parser.parse_args()

    env_path = load_env()
    if env_path:
        print(f"Loaded .env: {env_path}", flush=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Please set it or provide a .env file.")
    md_path = Path(args.md)
    md_text = read_text(md_path)

    # 1) select figures
    print("Selecting figures...", flush=True)
    sel = run_select_figures(md_path, args.model)
    figures = sel.get("figures") or []
    if not figures:
        print(json.dumps({"captions": [], "note": "No figures selected."}, ensure_ascii=False))
        return

    outputs = []
    for fig in figures[:3]:
        print(f"Preparing figure {fig}...", flush=True)
        context = extract_context_for_figure(md_text, str(fig))
        image_paths = find_figure_images(md_path, md_text, str(fig))
        if image_paths:
            print(f"Found images for Fig {fig}: {[str(p) for p in image_paths]}", flush=True)
        else:
            print(f"No images found near Fig {fig}. Proceeding with context only.", flush=True)
        user_prompt = build_caption_prompt(str(fig), context, args.max_captions)
        system_prompt = "You are a precise assistant that outputs strict JSON only."
        print(f"Calling OpenAI for Fig {fig}...", flush=True)
        obj = call_openai(args.model, system_prompt, user_prompt, images=image_paths, timeout=args.timeout)
        # normalize
        outputs.append({
            "figure": str(fig),
            "rubric": obj.get("rubric", {}),
            "captions": obj.get("captions", []) if args.max_captions <= 0 else obj.get("captions", [])[: args.max_captions],
            "reconstruction_prompt": obj.get("reconstruction_prompt", ""),
            "context_preview": context[:400],
            "images": [str(p) for p in image_paths]
        })

    print(json.dumps({"figures": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    main()


