#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List


def read_flist_range(flist_path: Path, pdf_root: Path, start: int, end: int) -> List[Path]:
    """Read PDF paths from flist file within specified range (1-based, inclusive)"""
    if not flist_path.exists():
        raise FileNotFoundError(f"Flist file not found: {flist_path}")
    
    pdf_paths = []
    print(f"Reading flist lines {start} to {end}...")
    
    with open(flist_path, 'r') as f:
        for idx, line in enumerate(f, 1):
            # Skip lines before start
            if idx < start:
                continue
            # Stop after end
            if idx > end:
                break
            
            line = line.strip()
            if line:
                # Convert relative path to absolute path
                pdf_path = pdf_root / line
                pdf_paths.append(pdf_path)
    
    if not pdf_paths:
        raise FileNotFoundError(f"No valid PDFs found from flist lines {start}-{end}")
    
    print(f"Loaded {len(pdf_paths)} PDFs from flist (lines {start}-{end})")
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
            print(f"\n{'='*60}")
            print(f"[{i}/{len(remaining_pdfs)}] Processing {pdf_path.name}...")
            print(f"{'='*60}")
            
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
            
            print(f"Running: {' '.join(cmd)}")
            
            # Don't capture output - let it display in real-time
            result = subprocess.run(cmd)
            
            # Check if output file was actually created
            pdf_stem = pdf_path.stem
            expected_md_file = out_dir / pdf_stem / "vlm" / f"{pdf_stem}.md"
            
            if result.returncode != 0:
                print(f"\n❌ ERROR: MinerU CLI failed for {pdf_path.name} (exit code: {result.returncode})", file=sys.stderr)
                failures += 1
            elif not expected_md_file.exists():
                print(f"\n⚠️  WARNING: MinerU returned success but output file not found: {expected_md_file}", file=sys.stderr)
                failures += 1
            else:
                file_size = expected_md_file.stat().st_size
                print(f"\n✅ SUCCESS: {pdf_path.name} -> {expected_md_file} ({file_size} bytes)")
                sys.stdout.flush()  # Flush output for real-time monitoring
        
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
    default_out_dir = Path("/blob/yasheng/arxiv_dataset/md_batch2/")
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

    # Validate range
    if args.start < 1 or args.end < args.start:
        print(f"Invalid range: start={args.start}, end={args.end}", file=sys.stderr)
        sys.exit(2)

    # Read only the specified range from flist (much faster for large files)
    sel = read_flist_range(flist_path, root_dir, args.start, args.end)
    print(f"Processing {len(sel)} PDFs directly without temporary directory")

    # Process the selected PDFs
    failures = process_pdfs_batch(sel, out_dir)

    if failures:
        print(f"Done with {failures} failures.")
        sys.exit(1)
    print("All done.")


if __name__ == "__main__":
    main()

