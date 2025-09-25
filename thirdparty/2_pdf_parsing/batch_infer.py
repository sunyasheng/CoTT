#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def find_pdfs(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"PDF root not found: {root_dir}")
    pdfs = sorted(root_dir.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under: {root_dir}")
    return pdfs


def run_mineru(pdf_path: Path, out_dir: Path, backend: str, device: str,
               max_num_seqs: int, max_model_len: int, extra_args: List[str]) -> int:
    cmd = [
        "mineru",
        "-p", str(pdf_path),
        "-o", str(out_dir),
        "--backend", backend,
        "--device", device,
        "--max-num-seqs", str(max_num_seqs),
        "--max-model-len", str(max_model_len),
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    default_pdf_root = Path("/ibex/user/suny0a/arxiv_dataset/pdf/")
    default_out_dir = Path("/ibex/user/suny0a/arxiv_dataset/md/")

    parser = argparse.ArgumentParser(description="Batch run MinerU over a range of PDFs (by index order)")
    parser.add_argument("--root", type=str, default=str(default_pdf_root),
                        help="Root directory containing PDFs (searched recursively)")
    parser.add_argument("--start", type=int, required=True,
                        help="1-based start index in the sorted PDF list (inclusive)")
    parser.add_argument("--end", type=int, required=True,
                        help="1-based end index in the sorted PDF list (inclusive)")
    parser.add_argument("--outdir", type=str, default=str(default_out_dir),
                        help="Output directory for MinerU outputs")
    parser.add_argument("--backend", type=str, default="vlm-vllm-engine",
                        help="MinerU backend (default: vlm-vllm-engine)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for MinerU (e.g., cuda or cpu)")
    parser.add_argument("--max-num-seqs", type=int, default=8,
                        help="vLLM max number of sequences")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="vLLM max model length")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="Any extra args to append to the mineru command (use after --)")

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    out_dir = Path(args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = find_pdfs(root_dir)

    if args.start < 1 or args.end < args.start or args.end > len(pdfs):
        print(f"Invalid range: start={args.start}, end={args.end}, total={len(pdfs)}", file=sys.stderr)
        sys.exit(2)

    sel = pdfs[args.start - 1: args.end]
    print(f"Found {len(pdfs)} PDFs; processing indices [{args.start}..{args.end}] -> {len(sel)} files")

    failures = 0
    for idx, pdf in enumerate(sel, start=args.start):
        print(f"[{idx}] Running MinerU on: {pdf}")
        per_pdf_out = out_dir / pdf.stem
        per_pdf_out.mkdir(parents=True, exist_ok=True)
        code = run_mineru(
            pdf_path=pdf,
            out_dir=per_pdf_out,
            backend=args.backend,
            device=args.device,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            extra_args=args.extra,
        )
        if code != 0:
            print(f"MinerU failed for {pdf} (exit={code})", file=sys.stderr)
            failures += 1

    if failures:
        print(f"Done with {failures} failures.")
        sys.exit(1)
    print("All done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def find_pdfs(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"PDF root not found: {root_dir}")
    pdfs = sorted(root_dir.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under: {root_dir}")
    return pdfs


def run_mineru(pdf_path: Path, out_dir: Path, backend: str, device: str,
               max_num_seqs: int, max_model_len: int, extra_args: List[str]) -> int:
    cmd = [
        "mineru",
        "-p", str(pdf_path),
        "-o", str(out_dir),
        "--backend", backend,
        "--device", device,
        "--max-num-seqs", str(max_num_seqs),
        "--max-model-len", str(max_model_len),
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    default_pdf_root = Path("/ibex/user/suny0a/arxiv_dataset/pdf/")
    default_out_dir = Path("/ibex/user/suny0a/arxiv_dataset/md/")

    parser = argparse.ArgumentParser(description="Batch run MinerU over a range of PDFs (by index order)")
    parser.add_argument("--root", type=str, default=str(default_pdf_root),
                        help="Root directory containing PDFs (searched recursively)")
    parser.add_argument("--start", type=int, required=True,
                        help="1-based start index in the sorted PDF list (inclusive)")
    parser.add_argument("--end", type=int, required=True,
                        help="1-based end index in the sorted PDF list (inclusive)")
    parser.add_argument("--outdir", type=str, default=str(default_out_dir),
                        help="Output directory for MinerU outputs")
    parser.add_argument("--backend", type=str, default="vlm-vllm-engine",
                        help="MinerU backend (default: vlm-vllm-engine)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for MinerU (e.g., cuda or cpu)")
    parser.add_argument("--max-num-seqs", type=int, default=8,
                        help="vLLM max number of sequences")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="vLLM max model length")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="Any extra args to append to the mineru command (use after --)")

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    out_dir = Path(args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = find_pdfs(root_dir)

    if args.start < 1 or args.end < args.start or args.end > len(pdfs):
        print(f"Invalid range: start={args.start}, end={args.end}, total={len(pdfs)}", file=sys.stderr)
        sys.exit(2)

    # Convert to 0-based slice
    sel = pdfs[args.start - 1: args.end]
    print(f"Found {len(pdfs)} PDFs; processing indices [{args.start}..{args.end}] -> {len(sel)} files")

    failures = 0
    for idx, pdf in enumerate(sel, start=args.start):
        print(f"[{idx}] Running MinerU on: {pdf}")
        per_pdf_out = out_dir / pdf.stem
        per_pdf_out.mkdir(parents=True, exist_ok=True)
        code = run_mineru(
            pdf_path=pdf,
            out_dir=per_pdf_out,
            backend=args.backend,
            device=args.device,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            extra_args=args.extra,
        )
        if code != 0:
            print(f"MinerU failed for {pdf} (exit={code})", file=sys.stderr)
            failures += 1

    if failures:
        print(f"Done with {failures} failures.")
        sys.exit(1)
    print("All done.")


if __name__ == "__main__":
    main()


