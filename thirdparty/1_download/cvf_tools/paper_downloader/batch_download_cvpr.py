#!/usr/bin/env python3
"""
Batch download CVPR papers Python script
Supports parallel downloading of papers from multiple years
"""

import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add lib directory to Python path
root_folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)

from code.paper_downloader_CVF import download_paper

def download_single_year(year, save_dir, download_main=True, download_workshops=True, 
                        download_supplement=False, time_step=5, downloader=None):
    """Download CVPR papers for a single year"""
    print(f"Starting download for CVPR {year}...")
    
    try:
        download_paper(
            year=year,
            conference='CVPR',
            save_dir=f"{save_dir}/CVPR_{year}",
            is_download_main_paper=download_main,
            is_download_supplement=download_supplement,
            time_step_in_seconds=time_step,
            is_download_main_conference=download_main,
            is_download_workshops=download_workshops,
            downloader=downloader
        )
        print(f"✅ CVPR {year} download completed")
        return True
    except Exception as e:
        print(f"❌ CVPR {year} download failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Batch download CVPR papers')
    parser.add_argument('--start-year', type=int, default=2013, help='Start year (default: 2013)')
    parser.add_argument('--end-year', type=int, default=2025, help='End year (default: 2025)')
    parser.add_argument('--save-dir', type=str, default='/ibex/user/suny0a/cvf_dataset/pdf', 
                       help='Save directory (default: /ibex/user/suny0a/cvf_dataset/pdf)')
    parser.add_argument('--max-workers', type=int, default=3, 
                       help='Maximum concurrent downloads (default: 3)')
    parser.add_argument('--download-main', action='store_true', default=True,
                       help='Download main conference papers')
    parser.add_argument('--download-workshops', action='store_true', default=True,
                       help='Download workshop papers')
    parser.add_argument('--download-supplement', action='store_true', default=False,
                       help='Download supplemental materials (default: False)')
    parser.add_argument('--time-step', type=int, default=5,
                       help='Download interval in seconds (default: 5)')
    parser.add_argument('--downloader', type=str, choices=['requests', 'IDM'], 
                       default='requests', help='Downloader type')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Prepare year list
    years = list(range(args.start_year, args.end_year + 1))
    
    print(f"Preparing to download CVPR {args.start_year}-{args.end_year} papers")
    print(f"Save directory: {args.save_dir}")
    print(f"Concurrency: {args.max_workers}")
    print(f"Download main conference: {args.download_main}")
    print(f"Download workshops: {args.download_workshops}")
    print(f"Download supplement: {args.download_supplement}")
    print(f"Download interval: {args.time_step} seconds")
    print("=" * 50)
    
    # Use thread pool for parallel downloading
    success_count = 0
    total_count = len(years)
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all download tasks
        future_to_year = {
            executor.submit(
                download_single_year,
                year,
                args.save_dir,
                args.download_main,
                args.download_workshops,
                args.download_supplement,
                args.time_step,
                None if args.downloader == 'requests' else 'IDM'
            ): year for year in years
        }
        
        # Wait for completion and show progress
        for future in as_completed(future_to_year):
            year = future_to_year[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                print(f"Progress: {success_count}/{total_count} completed")
            except Exception as e:
                print(f"❌ CVPR {year} exception occurred: {str(e)}")
    
    print("=" * 50)
    print(f"Download completed! Success: {success_count}/{total_count}")
    
    if success_count < total_count:
        print("Some years failed to download, please check logs and retry")

if __name__ == '__main__':
    main()
