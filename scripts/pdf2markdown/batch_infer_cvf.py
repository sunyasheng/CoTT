#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
import os
from pathlib import Path
from typing import List


def find_cvf_pdfs(root_dir: Path, dedupe_versions: bool = True, shuffle: bool = True) -> List[Path]:
    """Find CVF PDFs with conference-specific deduplication logic"""
    if not root_dir.exists():
        raise FileNotFoundError(f"CVF PDF root not found: {root_dir}")
    
    # Find all PDFs in CVF conference directories
    pdfs = []
    for conference_dir in root_dir.iterdir():
        if conference_dir.is_dir():
            # Look for PDFs in conference subdirectories (e.g., CVPR_2024/main_paper/)
            for subdir in conference_dir.rglob("*"):
                if subdir.is_dir() and any(subdir.name.startswith(prefix) for prefix in ["main_paper", "workshop"]):
                    pdfs.extend(sorted(subdir.glob("*.pdf")))
    
    if not pdfs:
        raise FileNotFoundError(f"No CVF PDFs found under: {root_dir}")
    
    print(f"Found {len(pdfs)} total CVF PDFs")
    
    if dedupe_versions:
        # Group by paper title (extract from filename like "paper-title_CVPR_2024.pdf")
        paper_groups = {}
        for pdf in pdfs:
            # Extract paper identifier from filename
            filename = pdf.name
            # Remove conference suffix and version info: "paper-title_CVPR_2024.pdf" -> "paper-title"
            base_name = re.sub(r'_(CVPR|ICCV|WACV|ACCV|ECCV)_\d{4}(\.pdf)?$', '', filename)
            base_name = re.sub(r'\.pdf$', '', base_name)
            
            if base_name not in paper_groups:
                paper_groups[base_name] = []
            paper_groups[base_name].append(pdf)
        
        # Keep only the first version of each paper
        deduped_pdfs = []
        for paper_id, versions in sorted(paper_groups.items()):
            # Sort versions and take the first one
            versions.sort()
            deduped_pdfs.append(versions[0])
        
        print(f"Deduplicated to {len(deduped_pdfs)} unique CVF papers")
        
        # Shuffle for better load balancing across shards
        if shuffle:
            import random
            random.seed(42)  # Fixed seed for reproducibility
            random.shuffle(deduped_pdfs)
            print(f"Shuffled CVF PDF list for better load balancing")
        
        return deduped_pdfs
    
    return pdfs


def process_cvf_pdfs_batch(pdf_paths: List[Path], out_dir: Path) -> int:
    """Process multiple CVF PDFs in a single MinerU CLI call to avoid model reloading"""
    try:
        # Filter out already processed PDFs
        remaining_pdfs = []
        skipped = 0
        
        for pdf_path in pdf_paths:
            # Check if corresponding markdown output already exists
            # PDF: /path/to/paper-title_CVPR_2024.pdf -> MD: /outdir/paper-title_CVPR_2024/vlm/paper-title_CVPR_2024.md
            pdf_stem = pdf_path.stem  # e.g., "paper-title_CVPR_2024"
            expected_md_dir = out_dir / pdf_stem / "vlm"
            expected_md_file = expected_md_dir / f"{pdf_stem}.md"
            
            if expected_md_file.exists():
                print(f"SKIP {pdf_path.name} -> {expected_md_file} (already processed)")
                skipped += 1
            else:
                remaining_pdfs.append(pdf_path)
        
        if not remaining_pdfs:
            print(f"All {len(pdf_paths)} CVF PDFs already processed, skipping batch")
            return 0
        
        print(f"Processing {len(remaining_pdfs)}/{len(pdf_paths)} CVF PDFs ({skipped} skipped)")
        
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
                "--max-num-seqs", "8",
                "--max-model-len", "12288",
                "--gpu-memory-utilization", "0.10"
            ]
            
            print(f"Processing batch of {len(remaining_pdfs)} CVF PDFs with single model load...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"MinerU CLI failed for CVF batch: {result.stderr}", file=sys.stderr)
                return len(remaining_pdfs)  # Remaining failed
            
            return 0  # All succeeded
        
    except Exception as e:
        print(f"Error processing CVF batch: {e}", file=sys.stderr)
        return len(pdf_paths)  # All failed


def main() -> None:
    # Set magic-pdf config path explicitly
    config_path = Path(__file__).parent / "magic-pdf.json"
    if config_path.exists():
        os.environ["MAGIC_PDF_CONFIG"] = str(config_path)
        print(f"Using config: {config_path}")
    else:
        print(f"Warning: Config file not found at {config_path}")
    
    default_pdf_root = Path("/ibex/user/suny0a/cvf_dataset/pdf/")
    default_out_dir = Path("/ibex/user/suny0a/cvf_dataset/md/")

    parser = argparse.ArgumentParser(description="Batch run MinerU over CVF PDFs using Python API (model loaded once)")
    parser.add_argument("--root", type=str, default=str(default_pdf_root),
                        help="Root directory containing CVF PDFs (searched recursively)")
    parser.add_argument("--start", type=int, required=True,
                        help="1-based start index in the sorted CVF PDF list (inclusive)")
    parser.add_argument("--end", type=int, required=True,
                        help="1-based end index in the sorted CVF PDF list (inclusive)")
    parser.add_argument("--outdir", type=str, default=str(default_out_dir),
                        help="Output directory for MinerU outputs")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable deduplication of paper versions")

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    out_dir = Path(args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = find_cvf_pdfs(root_dir, dedupe_versions=not args.no_dedupe)

    if args.start < 1 or args.end < args.start or args.end > len(pdfs):
        print(f"Invalid range: start={args.start}, end={args.end}, total={len(pdfs)}", file=sys.stderr)
        sys.exit(2)

    sel = pdfs[args.start - 1: args.end]
    print(f"Found {len(pdfs)} CVF PDFs; processing indices [{args.start}..{args.end}] -> {len(sel)} files")
    print("Using MinerU CLI with batch processing for CVF papers - model loaded once per batch")

    # Process all PDFs in a single batch to avoid model reloading
    failures = process_cvf_pdfs_batch(sel, out_dir)

    if failures:
        print(f"Done with {failures} failures.")
        sys.exit(1)
    print("All CVF papers done.")


if __name__ == "__main__":
    main()
