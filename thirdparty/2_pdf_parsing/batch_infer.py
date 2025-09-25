#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from typing import List

try:
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.enums import SupportedPdfParseMethod
except ImportError as e:
    print(f"Error importing magic_pdf modules: {e}")
    print("Please install MinerU with: pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com")
    sys.exit(1)


def find_pdfs(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"PDF root not found: {root_dir}")
    pdfs = sorted(root_dir.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under: {root_dir}")
    return pdfs


def process_pdf_with_api(pdf_path: Path, out_dir: Path) -> bool:
    """Process a single PDF using MinerU Python API"""
    try:
        name_without_suff = pdf_path.stem
        per_pdf_out = out_dir / name_without_suff
        per_pdf_out.mkdir(parents=True, exist_ok=True)
        
        # Set up directories
        local_image_dir = per_pdf_out / "images"
        local_md_dir = per_pdf_out
        local_image_dir.mkdir(exist_ok=True)
        
        # Initialize data readers/writers
        image_writer = FileBasedDataWriter(str(local_image_dir))
        md_writer = FileBasedDataWriter(str(local_md_dir))
        reader = FileBasedDataReader("")
        
        # Read PDF file
        pdf_bytes = reader.read(str(pdf_path))
        
        # Create dataset and process
        ds = PymuDocDataset(pdf_bytes)
        
        # Determine processing method and apply analysis
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)
        
        # Save results
        infer_result.draw_model(str(local_md_dir / f"{name_without_suff}_model.pdf"))
        
        md_content = pipe_result.get_markdown(str(local_image_dir))
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", str(local_image_dir))
        
        content_list_content = pipe_result.get_content_list(str(local_image_dir))
        pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", str(local_image_dir))
        
        return True
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}", file=sys.stderr)
        return False


def main() -> None:
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
    print("Using MinerU Python API - model will be loaded once and reused for all PDFs")

    failures = 0
    for idx, pdf in enumerate(sel, start=args.start):
        print(f"[{idx}] Processing: {pdf}")
        success = process_pdf_with_api(pdf, out_dir)
        if not success:
            failures += 1

    if failures:
        print(f"Done with {failures} failures.")
        sys.exit(1)
    print("All done.")


if __name__ == "__main__":
    main()