#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path


def load_env():
    try:
        from dotenv import load_dotenv  # optional
        load_dotenv()
    except Exception:
        pass


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


def call_qwen(model: str, system_prompt: str, user_prompt: str):
    try:
        from openai import OpenAI
        # OpenAI-compatible client; use QWEN_* or OPENAI_* envs
        base_url = os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Qwen call failed: {e}")


def build_prompt(candidates):
    guide = (
        "You are given snippets from a paper's markdown mentioning figures. "
        "Pick up to 3 figure numbers that most likely correspond to schematic diagrams, architecture/framework overviews, or process/flow charts. "
        "Exclude natural images/photographs and typical data charts (bar/line/scatter/histogram). "
        "If uncertain, skip. Return a strict JSON object with keys: figures (array of figure numbers as strings, max 3), rationale (short string)."
    )
    parts = ["Snippets:"]
    for c in candidates:
        parts.append(f"- line {c['line']}: fig(s) {', '.join(c['fig_nums'])}\n{c['snippet']}")
    return guide + "\n\n" + "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Select up to 3 framework/flow figures from markdown via Qwen2.5-VL (OpenAI-compatible API).")
    parser.add_argument("--md", default="debug/0806.1636v1/vlm/0806.1636v1.md", help="Path to markdown file")
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"), help="Qwen model name")
    args = parser.parse_args()

    load_env()
    md_path = Path(args.md)
    text = read_text(md_path)
    candidates = extract_figure_mentions(text)
    if not candidates:
        print(json.dumps({"figures": [], "rationale": "No figure mentions found in markdown."}, ensure_ascii=False))
        return

    system_prompt = "You are a precise assistant that outputs strict JSON only."
    user_prompt = build_prompt(candidates)
    content = call_qwen(args.model, system_prompt, user_prompt)

    json_text = None
    m = re.search(r"```\s*([a-zA-Z]+)?\s*\n([\s\S]*?)```", content)
    if m:
        json_text = m.group(2).strip()
    if not json_text:
        s = content.strip()
        i = s.find('{')
        j = s.rfind('}')
        if i != -1 and j != -1 and j > i:
            json_text = s[i:j+1]
    if not json_text:
        json_text = content

    try:
        obj = json.loads(json_text)
    except Exception:
        obj = {"figures": [], "rationale": f"Non-JSON reply: {content[:200]}..."}
    print(json.dumps(obj, ensure_ascii=False))


if __name__ == "__main__":
    main()


