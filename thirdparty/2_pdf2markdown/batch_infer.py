#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
import os
from pathlib import Path
from typing import List


def find_pdfs(root_dir: Path, dedupe_versions: bool = True, shuffle: bool = True) -> List[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"PDF root not found: {root_dir}")
    pdfs = sorted(root_dir.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under: {root_dir}")
    
    if dedupe_versions:
        # Group by paper ID (extract from path like /path/to/1905.12185v3.pdf)
        paper_groups = {}
        for pdf in pdfs:
            # Extract paper ID from filename: 1905.12185v3.pdf -> 1905.12185
            filename = pdf.name
            match = re.match(r'(\d{4}\.\d{4,5})', filename)
            if match:
                paper_id = match.group(1)
                if paper_id not in paper_groups:
                    paper_groups[paper_id] = []
                paper_groups[paper_id].append(pdf)
        
        # Keep only the first version of each paper
        deduped_pdfs = []
        for paper_id, versions in sorted(paper_groups.items()):
            # Sort versions and take the first one
            versions.sort()
            deduped_pdfs.append(versions[0])
        
        print(f"Found {len(pdfs)} total PDFs, deduplicated to {len(deduped_pdfs)} unique papers")
        
        # Shuffle for better load balancing across shards
        if shuffle:
            import random
            random.seed(42)  # Fixed seed for reproducibility
            random.shuffle(deduped_pdfs)
            print(f"Shuffled PDF list for better load balancing")
        
        return deduped_pdfs
    
    return pdfs


def process_pdfs_batch(pdf_paths: List[Path], out_dir: Path) -> int:
    """Process multiple PDFs in a single MinerU CLI call to avoid model reloading"""
    try:
        # Filter out already processed PDFs
        remaining_pdfs = []
        skipped = 0
        
        for pdf_path in pdf_paths:
            # Check if corresponding markdown output already exists
            # PDF: /path/to/1905.12185v3.pdf -> MD: /outdir/1905.12185v3/vlm/1905.12185v3.md
            pdf_stem = pdf_path.stem  # e.g., "1905.12185v3"
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
        
        # Create a temporary directory for this batch
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf_dir = Path(temp_dir) / "pdfs"
            temp_pdf_dir.mkdir()
            
            # Copy remaining PDFs to temp directory
            for pdf_path in remaining_pdfs:
                import shutil
                shutil.copy2(pdf_path, temp_pdf_dir / pdf_path.name)
            
            # Use MinerU CLI command on the directory
            cmd = [
                "mineru",
                "-p", str(temp_pdf_dir),
                "-o", str(out_dir),
                "--backend", "vlm-vllm-engine",
                "--device", "cuda",
                "--max-num-seqs", "8",  # 增加并发数，适合A100
                "--max-model-len", "12288",  # 保持原始值，避免bug
                "--gpu-memory-utilization", "0.9"  # 提高显存利用率，适合A100
            ]
            
            print(f"Processing batch of {len(remaining_pdfs)} PDFs with single model load...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"MinerU CLI failed for batch: {result.stderr}", file=sys.stderr)
                return len(remaining_pdfs)  # Remaining failed
            
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
    parser.add_argument("--no-dedupe", action="store_true", help="Disable deduplication of paper versions")
    # Note: backend, device, max-num-seqs, max-model-len are handled by MinerU's internal configuration
    # The Python API doesn't expose these vLLM parameters directly

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    out_dir = Path(args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = find_pdfs(root_dir, dedupe_versions=not args.no_dedupe)

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