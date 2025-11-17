#!/usr/bin/env python3
"""
Download PDFs from OpenReview for ICLR papers.
Reads parquet file and downloads PDFs based on paper IDs.
"""

import os
import sys
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import time
from urllib.parse import quote

### download pdfs: https://raw.githubusercontent.com/berenslab/iclr-dataset/refs/heads/main/data/iclr26v1.parquet
def get_pdf_url(paper_id: str) -> str:
    """
    Construct OpenReview PDF URL from paper ID.
    OpenReview PDF URLs are typically: https://openreview.net/pdf?id={paper_id}
    """
    # URL encode the paper ID
    encoded_id = quote(paper_id, safe='')
    return f"https://openreview.net/pdf?id={encoded_id}"


def download_pdf(paper_id: str, year: int, output_dir: str, session: requests.Session, 
                 timeout: int = 30, max_retries: int = 3) -> tuple[str, bool, str]:
    """
    Download a PDF from OpenReview.
    
    Returns:
        (output_path, success, error_message)
    """
    pdf_url = get_pdf_url(paper_id)
    output_path = os.path.join(output_dir, f"{year}", f"{paper_id}.pdf")
    
    # Create year directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Skip if file already exists and has content
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path, True, "already_exists"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/pdf,*/*",
    }
    
    for attempt in range(max_retries):
        try:
            response = session.get(pdf_url, headers=headers, timeout=timeout, stream=True)
            
            if response.status_code == 404:
                return output_path, False, f"404 Not Found"
            
            if response.status_code != 200:
                return output_path, False, f"HTTP {response.status_code}"
            
            # Check if response is actually a PDF
            content_type = response.headers.get("Content-Type", "").lower()
            
            # Download PDF to temporary file
            tmp_path = output_path + ".tmp"
            first_bytes = b""
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        if not first_bytes and len(chunk) >= 8:
                            first_bytes = chunk[:8]
                        f.write(chunk)
            
            # Validate PDF
            if not first_bytes.startswith(b"%PDF-") and "application/pdf" not in content_type:
                os.remove(tmp_path)
                return output_path, False, f"Not a PDF (Content-Type: {content_type})"
            
            # Validate PDF
            file_size = os.path.getsize(tmp_path)
            if file_size < 1000:  # PDFs should be at least 1KB
                os.remove(tmp_path)
                return output_path, False, f"File too small ({file_size} bytes)"
            
            # Move temp file to final location
            os.replace(tmp_path, output_path)
            return output_path, True, "success"
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return output_path, False, "Timeout"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return output_path, False, f"Request error: {str(e)}"
        except Exception as e:
            return output_path, False, f"Unexpected error: {str(e)}"
    
    return output_path, False, "Max retries exceeded"


def main():
    # Paths
    parquet_path = Path(__file__).parent.parent.parent / "debug" / "iclr2017_2026" / "iclr26v1.parquet"
    output_base_dir = Path("/blob/yasheng/iclr_dataset")
    
    print(f"Loading parquet file: {parquet_path}")
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}")
        sys.exit(1)
    
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} papers from parquet file")
    
    # Create output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by year and download all papers
    downloaded = 0
    failed = 0
    skipped = 0
    
    print(f"\nDownloading ALL papers from parquet file...")
    print("=" * 70)
    
    with requests.Session() as session:
        total_papers = len(df)
        overall_progress = tqdm(total=total_papers, desc="Overall Progress", position=0, leave=True)
        
        for year in sorted(df['year'].unique()):
            year_papers = df[df['year'] == year]
            year_count = len(year_papers)
            
            # Use tqdm for progress bar with detailed info
            year_progress = tqdm(
                year_papers.iterrows(), 
                total=year_count, 
                desc=f"Year {year:4d}",
                position=1,
                leave=False,
                unit="paper",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
            for idx, row in year_progress:
                paper_id = row['id']
                
                output_path, success, message = download_pdf(
                    paper_id, year, str(output_base_dir), session
                )
                
                if success:
                    if message == "already_exists":
                        skipped += 1
                    else:
                        downloaded += 1
                else:
                    failed += 1
                    # Only print failures to reduce output
                    if failed <= 10 or failed % 100 == 0:
                        year_progress.write(f"  ✗ {paper_id}: Failed - {message}")
                
                # Update progress bars
                overall_progress.update(1)
                year_progress.set_postfix({
                    'D': downloaded, 
                    'S': skipped, 
                    'F': failed
                })
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            
            year_progress.close()
        
        overall_progress.close()
    
    print("\n" + "=" * 70)
    print(f"Summary:")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {downloaded + skipped + failed}")
    print(f"\nPDFs saved to: {output_base_dir}")


if __name__ == "__main__":
    main()

