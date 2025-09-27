#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def find_mds(root_dir: Path, pattern: str = "*.md") -> list[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"MD root not found: {root_dir}")
    mds = sorted(root_dir.rglob(pattern))
    if not mds:
        raise FileNotFoundError(f"No markdown files found under: {root_dir} (pattern={pattern})")
    return mds


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run qwen_select_and_caption over a slice of markdowns (indices)")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing markdown outputs (searched recursively)")
    parser.add_argument("--start", type=int, required=True, help="1-based start index in the sorted MD list (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="1-based end index in the sorted MD list (inclusive)")
    parser.add_argument("--glob", type=str, default="*.md", help="Glob when scanning root (e.g. '*/vlm/*.md')")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct", help="Qwen VL model")
    parser.add_argument("--max_figures", type=int, default=3, help="Max figures per paper")
    parser.add_argument("--max_captions", type=int, default=0, help="Max steps per figure (0 = unlimited)")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout seconds")
    parser.add_argument("--outdir", type=str, default="", help="Write each result JSON to this directory")
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    md_files = find_mds(root_dir, args.glob)

    if args.start < 1 or args.end < args.start or args.end > len(md_files):
        print(f"Invalid range: start={args.start}, end={args.end}, total={len(md_files)}", file=sys.stderr)
        sys.exit(2)

    sel = md_files[args.start - 1: args.end]
    print(f"Found {len(md_files)} MD files; processing indices [{args.start}..{args.end}] -> {len(sel)} files")

    script = Path(__file__).parent / "qwen_select_and_caption.py"
    if not script.exists():
        raise FileNotFoundError(f"qwen_select_and_caption.py not found at {script}")

    failures = 0
    skipped = 0
    for md in sel:
        # Check if output already exists
        if args.outdir:
            out_dir = Path(args.outdir).expanduser().resolve()
            output_file = out_dir / f"{md.stem}.json"
            if output_file.exists():
                print(f"SKIP {md} -> {output_file} (already exists)")
                skipped += 1
                continue
        
        cmd = [
            "python3", str(script),
            "--md", str(md),
            "--model", args.model,
            "--max_figures", str(args.max_figures),
            "--max_captions", str(args.max_captions),
            "--timeout", str(args.timeout),
        ]
        if args.outdir:
            cmd += ["--outdir", str(Path(args.outdir).expanduser().resolve())]
        print(f"Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAILED {md}: {res.stderr}", file=sys.stderr)
            failures += 1
        else:
            print(res.stdout)

    if failures:
        print(f"Done with {failures} failures, {skipped} skipped.")
        sys.exit(1)
    print(f"All done. {skipped} files skipped (already processed).")


if __name__ == "__main__":
    main()


