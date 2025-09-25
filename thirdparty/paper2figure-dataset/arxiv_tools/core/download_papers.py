#!/usr/bin/env python3
"""
Core arXiv PDF Downloader
Downloads PDFs from Google Cloud Storage based on paper IDs
"""

import os
import sys
import argparse
from pathlib import Path
from google.cloud import storage
from tqdm import tqdm

def download_papers(paper_ids, output_dir, project_id=None, bucket_name="arxiv-dataset", prefix="arxiv/"):
    """Download papers from Google Cloud Storage

    Args:
        paper_ids: Iterable of arXiv IDs without version (e.g., "0704.0001")
        output_dir: Directory to save PDFs into
        project_id: Optional GCP project id for the Storage client
        bucket_name: GCS bucket name
        prefix: Object key prefix inside the bucket (e.g., "arxiv/" or "paper2figure_dataset/pdf/")
    """
    
    # Initialize GCS client
    try:
        if project_id:
            client = storage.Client(project=project_id)
        else:
            client = storage.Client()
    except Exception as e:
        print(f"❌ Error initializing Google Cloud client: {e}")
        print("Make sure you have Google Cloud credentials set up")
        sys.exit(1)
    
    bucket = client.bucket(bucket_name)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📄 Downloading {len(paper_ids)} papers to: {output_path.absolute()}")
    
    downloaded = 0
    failed = 0
    
    for paper_id in tqdm(paper_ids, desc="Downloading"):
        try:
            # Try different version patterns
            versions = [f"{paper_id}v1", f"{paper_id}v2", f"{paper_id}v3", f"{paper_id}v4", f"{paper_id}v5"]
            
            downloaded_file = None
            for version in versions:
                blob_name = f"{prefix}{version}.pdf"
                blob = bucket.blob(blob_name)
                
                if blob.exists():
                    # Create paper directory
                    paper_dir = output_path / paper_id
                    paper_dir.mkdir(exist_ok=True)
                    
                    # Download file
                    output_file = paper_dir / f"{version}.pdf"
                    blob.download_to_filename(str(output_file))
                    downloaded_file = output_file
                    break
            
            if downloaded_file:
                downloaded += 1
            else:
                failed += 1
                print(f"\n❌ Not found: {paper_id}")
                
        except Exception as e:
            failed += 1
            print(f"\n❌ Error downloading {paper_id}: {e}")
    
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successfully downloaded: {downloaded}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 Output directory: {output_path.absolute()}")

def main():
    parser = argparse.ArgumentParser(description="Download arXiv PDFs from Google Cloud Storage")
    parser.add_argument("--papers", "-p", required=True, help="Paper IDs (comma-separated or file path)")
    parser.add_argument("--output", "-o", default="./pdfs", help="Output directory for PDFs")
    parser.add_argument("--project", help="Google Cloud Project ID (optional)")
    parser.add_argument("--bucket", default="arxiv-dataset", help="GCS bucket name")
    parser.add_argument("--prefix", default="arxiv/", help="GCS object prefix (e.g., 'arxiv/' or 'paper2figure_dataset/pdf/')")
    
    args = parser.parse_args()
    
    # Parse paper IDs
    if os.path.exists(args.papers):
        with open(args.papers, 'r') as f:
            paper_ids = [line.strip() for line in f if line.strip()]
    else:
        paper_ids = [pid.strip() for pid in args.papers.split(',')]
    
    download_papers(paper_ids, args.output, args.project, args.bucket, args.prefix)

if __name__ == "__main__":
    main()
