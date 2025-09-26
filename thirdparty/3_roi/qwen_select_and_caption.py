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


def build_selection_prompt(candidates, max_figures: int) -> str:
    guide = (
        "You are given snippets from a paper's markdown mentioning figures. "
        f"Pick up to {max_figures} figure numbers that most likely correspond to schematic diagrams, architecture/framework overviews, or process/flow charts. "
        "Exclude natural images/photographs and typical data charts (bar/line/scatter/histogram). "
        "If uncertain, skip. Return a strict JSON object with keys: figures (array of figure numbers as strings, max N), rationale (short string)."
    )
    parts = ["Snippets:"]
    for c in candidates:
        parts.append(f"- line {c['line']}: fig(s) {', '.join(c['fig_nums'])}\n{c['snippet']}")
    return guide + "\n\n" + "\n\n".join(parts)


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


class QwenOpenAIClient:
    def __init__(self, timeout: int = 120):
        from openai import OpenAI
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        base_url = os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        else:
            self.client = OpenAI(api_key=api_key, timeout=timeout)

    def chat_json(self, model: str, system_prompt: str, user_prompt: str, image_paths: list[Path] | None = None) -> dict:
        user_content = [{"type": "text", "text": user_prompt}]
        if image_paths:
            for p in image_paths:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": encode_image_to_data_url(p)}
                })
        resp = self.client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser(description="Select figures and generate rubric + captions in one pass (Qwen OpenAI-compatible)")
    parser.add_argument("--md", required=True, help="Path to the paper markdown")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Qwen VL model name")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout seconds")
    parser.add_argument("--max_figures", type=int, default=3, help="Max figures to process")
    parser.add_argument("--max_captions", type=int, default=3, help="Max captions per figure (set 0 for unlimited)")
    parser.add_argument("--outdir", type=str, default="", help="If set, write JSON to this directory as <md_stem>.json")
    args = parser.parse_args()

    env_path = load_env()
    if env_path:
        print(f"Loaded .env: {env_path}")

    md_path = Path(args.md)
    md_text = read_text(md_path)

    # 1) select figures
    print("Selecting figures...", flush=True)
    candidates = extract_figure_mentions(md_text)
    if not candidates:
        print(json.dumps({"figures": [], "note": "No figure mentions found in markdown."}, ensure_ascii=False))
        return
    system_prompt_sel = "You are a precise assistant that outputs strict JSON only."
    user_prompt_sel = build_selection_prompt(candidates, args.max_figures)
    qclient = QwenOpenAIClient(timeout=args.timeout)
    sel_obj = qclient.chat_json(args.model, system_prompt_sel, user_prompt_sel, image_paths=None)
    figures = sel_obj.get("figures") or []
    if not figures:
        print(json.dumps({"figures": [], "note": "No figures selected."}, ensure_ascii=False))
        return

    # 2) caption each figure
    outputs = []
    for fig in figures[: args.max_figures]:
        print(f"Preparing figure {fig}...", flush=True)
        context = extract_context_for_figure(md_text, str(fig))
        image_paths = find_figure_images(md_path, md_text, str(fig))
        if image_paths:
            print(f"Found images for Fig {fig}: {[str(p) for p in image_paths]}")
        else:
            print(f"No images found near Fig {fig}. Proceeding with context only.")
        user_prompt_cap = build_caption_prompt(str(fig), context, args.max_captions)
        system_prompt_cap = "You are a precise assistant that outputs strict JSON only."
        obj = qclient.chat_json(args.model, system_prompt_cap, user_prompt_cap, image_paths=image_paths)
        outputs.append({
            "figure": str(fig),
            "rubric": obj.get("rubric", {}),
            "captions": obj.get("captions", []) if args.max_captions <= 0 else obj.get("captions", [])[: args.max_captions],
            "reconstruction_prompt": obj.get("reconstruction_prompt", ""),
            "context_preview": context[:400],
            "images": [str(p) for p in image_paths],
        })

    result = {"figures": outputs}
    if args.outdir:
        out_dir = Path(args.outdir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = md_path.stem
        out_path = out_dir / f"{stem}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(out_path))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()


# Example:
# export QWEN_BASE_URL=http://10.109.17.91:22002/v1
# export QWEN_API_KEY=EMPTY
# python3 thirdparty/3_roi/qwen_select_and_caption.py \
#   --md debug/0806.1636v1/vlm/0806.1636v1.md \
#   --model Qwen/Qwen2.5-VL-32B-Instruct \
#   --max_figures 3 \
#   --max_captions 0 \
#   --timeout 120


