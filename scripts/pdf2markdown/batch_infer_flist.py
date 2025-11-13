#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List


def read_flist(flist_path: Path, pdf_root: Path) -> List[Path]:
    """Read PDF paths from flist file and convert to absolute paths"""
    if not flist_path.exists():
        raise FileNotFoundError(f"Flist file not found: {flist_path}")
    
    pdf_paths = []
    with open(flist_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Convert relative path to absolute path
                pdf_path = pdf_root / line
                if pdf_path.exists():
                    pdf_paths.append(pdf_path)
                else:
                    print(f"Warning: PDF not found: {pdf_path}")
    
    if not pdf_paths:
        raise FileNotFoundError(f"No valid PDFs found from flist: {flist_path}")
    
    print(f"Loaded {len(pdf_paths)} PDFs from flist")
    return pdf_paths


def process_pdfs_batch(pdf_paths: List[Path], out_dir: Path) -> int:
    """Process multiple PDFs in a single MinerU CLI call to avoid model reloading"""
    try:
        # Filter out already processed PDFs
        remaining_pdfs = []
        skipped = 0
        
        for pdf_path in pdf_paths:
            # Check if corresponding markdown output already exists
            # PDF: /path/to/2104.03057/2104.03057v1.pdf -> MD: /outdir/2104.03057v1/vlm/2104.03057v1.md
            pdf_stem = pdf_path.stem  # e.g., "2104.03057v1"
            expected_md_dir = out_dir / pdf_stem / "vlm"
            expected_md_file = expected_md_dir / f"{pdf_stem}.md"
            
            if expected_md_file.exists():
                print(f"SKIP {pdf_path.name} -> {expected_md_file} (already processed)")
                skipped += 1
            else:
                remaining_pdfs.append(pdf_path)
        
        if not remaining_pdfs:
            print(f"All {len(pdf_paths)} PDFs already processed, skipping batch")
            return 0
        
        print(f"Processing {len(remaining_pdfs)}/{len(pdf_paths)} PDFs ({skipped} skipped)")
        
        # Process PDFs directly without copying to temp directory
        # Call MinerU on each PDF individually to avoid issues with different directory structures
        failures = 0
        for i, pdf_path in enumerate(remaining_pdfs, 1):
            print(f"[{i}/{len(remaining_pdfs)}] Processing {pdf_path.name}...")
            
            cmd = [
                "mineru",
                "-p", str(pdf_path),
                "-o", str(out_dir),
                "--backend", "vlm-vllm-engine",
                "--device", "cuda",
                "--max-num-seqs", "8",  # 增加并发数，适合A100
                "--max-model-len", "12288",  # 保持原始值，避免bug
                "--gpu-memory-utilization", "0.9",  # 提高显存利用率，适合A100
                "--f_draw_layout_bbox", "False",  # 禁用 layout PDF 生成
                "--f_dump_orig_pdf", "False"  # 禁用 original PDF 生成
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"ERROR: MinerU CLI failed for {pdf_path.name}: {result.stderr}", file=sys.stderr)
                failures += 1
            else:
                print(f"SUCCESS: {pdf_path.name}")
        
        return failures
        
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
    
    default_pdf_root = Path("/blob/yasheng/arxiv_dataset/pdf/")
    default_out_dir = Path("/blob/yasheng/arxiv_dataset/md/")
    default_flist = Path(__file__).parent.parent / "flists" / "arxiv_pdf_remaining.list"

    parser = argparse.ArgumentParser(description="Batch run MinerU over PDFs from a flist file")
    parser.add_argument("--flist", type=str, default=str(default_flist),
                        help="Path to flist file containing relative PDF paths")
    parser.add_argument("--root", type=str, default=str(default_pdf_root),
                        help="Root directory for PDF paths (flist paths are relative to this)")
    parser.add_argument("--start", type=int, required=True,
                        help="1-based start index in the flist (inclusive)")
    parser.add_argument("--end", type=int, required=True,
                        help="1-based end index in the flist (inclusive)")
    parser.add_argument("--outdir", type=str, default=str(default_out_dir),
                        help="Output directory for MinerU outputs")

    args = parser.parse_args()

    flist_path = Path(args.flist).resolve()
    root_dir = Path(args.root).resolve()
    out_dir = Path(args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read all PDFs from flist
    all_pdfs = read_flist(flist_path, root_dir)

    if args.start < 1 or args.end < args.start or args.end > len(all_pdfs):
        print(f"Invalid range: start={args.start}, end={args.end}, total={len(all_pdfs)}", file=sys.stderr)
        sys.exit(2)

    # Select the range
    sel = all_pdfs[args.start - 1: args.end]
    print(f"Processing PDFs from flist: indices [{args.start}..{args.end}] -> {len(sel)} files")
    print("Processing PDFs directly without temporary directory")

    # Process the selected PDFs
    failures = process_pdfs_batch(sel, out_dir)

    if failures:
        print(f"Done with {failures} failures.")
        sys.exit(1)
    print("All done.")


if __name__ == "__main__":
    main()

