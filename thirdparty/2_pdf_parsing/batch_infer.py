#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List


def find_pdfs(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"PDF root not found: {root_dir}")
    pdfs = sorted(root_dir.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under: {root_dir}")
    return pdfs


def process_pdfs_batch(pdf_paths: List[Path], out_dir: Path) -> int:
    """Process multiple PDFs in a single MinerU CLI call to avoid model reloading"""
    try:
        # Create a temporary directory for this batch
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf_dir = Path(temp_dir) / "pdfs"
            temp_pdf_dir.mkdir()
            
            # Copy PDFs to temp directory
            for pdf_path in pdf_paths:
                import shutil
                shutil.copy2(pdf_path, temp_pdf_dir / pdf_path.name)
            
            # Use MinerU CLI command on the directory
            cmd = [
                "mineru",
                "-p", str(temp_pdf_dir),
                "-o", str(out_dir),
                "--backend", "vlm-vllm-engine",
                "--device", "cuda",
                "--max-num-seqs", "8",
                "--max-model-len", "12288",
                "--gpu-memory-utilization", "0.10"
            ]
            
            print(f"Processing batch of {len(pdf_paths)} PDFs with single model load...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"MinerU CLI failed for batch: {result.stderr}", file=sys.stderr)
                return len(pdf_paths)  # All failed
            
            return 0  # All succeeded
        
    except Exception as e:
        print(f"Error processing batch: {e}", file=sys.stderr)
        return len(pdf_paths)  # All failed


def main() -> None:
    # Set magic-pdf config path explicitly
    config_path = Path(__file__).parent / "magic-pdf.json"
    if config_path.exists():
        os.environ["MAGIC_PDF_CONFIG"] = str(config_path)
        print(f"Using config: {config_path}")
    else:
        print(f"Warning: Config file not found at {config_path}")
    
    default_pdf_root = Path("/ibex/user/suny0a/arxiv_dataset/pdf/")
    default_out_dir = Path("/ibex/user/suny0a/arxiv_dataset/md/")

    parser = argparse.ArgumentParser(description="Batch run MinerU over a range of PDFs using Python API (model loaded once)")
    parser.add_argument("--root", type=str, default=str(default_pdf_root),
                        help="Root directory containing PDFs (searched recursively)")
    parser.add_argument("--start", type=int, required=True,
                        help="1-based start index in the sorted PDF list (inclusive)")
    parser.add_argument("--end", type=int, required=True,
                        help="1-based end index in the sorted PDF list (inclusive)")
    parser.add_argument("--outdir", type=str, default=str(default_out_dir),
                        help="Output directory for MinerU outputs")
    # Note: backend, device, max-num-seqs, max-model-len are handled by MinerU's internal configuration
    # The Python API doesn't expose these vLLM parameters directly

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
    print("Using MinerU CLI with batch processing - model loaded once per batch")

    # Process all PDFs in a single batch to avoid model reloading
    failures = process_pdfs_batch(sel, out_dir)

    if failures:
        print(f"Done with {failures} failures.")
        sys.exit(1)
    print("All done.")


if __name__ == "__main__":
    main()