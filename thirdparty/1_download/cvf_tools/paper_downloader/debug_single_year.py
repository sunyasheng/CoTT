#!/usr/bin/env python3
"""
Debug script for downloading CVPR papers for a single year
Use this script to test and debug the download process on a single node
"""

import os
import sys
import argparse

# Add lib directory to Python path
root_folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)

from code.paper_downloader_CVF import download_paper, save_csv, download_from_csv

def debug_single_year(year, save_dir, download_main=True, download_workshops=True, 
                     download_supplement=False, time_step=5, downloader=None):
    """Debug download for a single year with detailed logging"""
    print(f"=== DEBUG: CVPR {year} Download ===")
    print(f"Save directory: {save_dir}")
    print(f"Download main conference: {download_main}")
    print(f"Download workshops: {download_workshops}")
    print(f"Download supplement: {download_supplement}")
    print(f"Time step: {time_step} seconds")
    print(f"Downloader: {downloader}")
    print("=" * 50)
    
    try:
        # Test CSV generation first
        print(f"Step 1: Generating CSV for CVPR {year}...")
        total_papers = save_csv(year, 'CVPR')
        print(f"✅ Found {total_papers} main conference papers")
        
        if download_workshops:
            print(f"Step 2: Generating CSV for CVPR {year} workshops...")
            from code.paper_downloader_CVF import save_csv_workshops
            total_workshops = save_csv_workshops(year, 'CVPR')
            print(f"✅ Found {total_workshops} workshop papers")
        
        # Test download
        print(f"Step 3: Starting download for CVPR {year}...")
        download_paper(
            year=year,
            conference='CVPR',
            save_dir=save_dir,
            is_download_main_paper=download_main,
            is_download_supplement=download_supplement,
            time_step_in_seconds=time_step,
            is_download_main_conference=download_main,
            is_download_workshops=download_workshops,
            downloader=downloader
        )
        
        print(f"✅ CVPR {year} download completed successfully!")
        
        # Check results
        main_dir = f"{save_dir}/CVPR_{year}/main_paper"
        workshop_dir = f"{save_dir}/CVPR_WS_{year}/main_paper"
        
        if os.path.exists(main_dir):
            main_count = len([f for f in os.listdir(main_dir) if f.endswith('.pdf')])
            print(f"📁 Main conference: {main_count} PDFs downloaded")
        
        if download_workshops and os.path.exists(workshop_dir):
            workshop_count = len([f for f in os.listdir(workshop_dir) if f.endswith('.pdf')])
            print(f"📁 Workshops: {workshop_count} PDFs downloaded")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Debug CVPR download for single year')
    parser.add_argument('--year', type=int, required=True, help='Year to download (e.g., 2024)')
    parser.add_argument('--save-dir', type=str, default='/ibex/user/suny0a/cvf_dataset/pdf', 
                       help='Save directory (default: /ibex/user/suny0a/cvf_dataset/pdf)')
    parser.add_argument('--download-main', action='store_true', default=True,
                       help='Download main conference papers (default: True)')
    parser.add_argument('--download-workshops', action='store_true', default=True,
                       help='Download workshop papers (default: True)')
    parser.add_argument('--download-supplement', action='store_true', default=False,
                       help='Download supplemental materials (default: False)')
    parser.add_argument('--time-step', type=int, default=5,
                       help='Download interval in seconds (default: 5)')
    parser.add_argument('--downloader', type=str, choices=['requests', 'IDM'], 
                       default='requests', help='Downloader type (default: requests)')
    parser.add_argument('--test-only', action='store_true', default=False,
                       help='Only test CSV generation, do not download')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Starting debug session for CVPR {args.year}")
    print(f"Save directory: {args.save_dir}")
    print(f"Test only: {args.test_only}")
    print("=" * 50)
    
    if args.test_only:
        print("Running in test mode (CSV generation only)...")
        try:
            total_papers = save_csv(args.year, 'CVPR')
            print(f"✅ Main conference: {total_papers} papers found")
            
            if args.download_workshops:
                from code.paper_downloader_CVF import save_csv_workshops
                total_workshops = save_csv_workshops(args.year, 'CVPR')
                print(f"✅ Workshops: {total_workshops} papers found")
            
            print("Test completed successfully!")
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        # Full download
        success = debug_single_year(
            year=args.year,
            save_dir=args.save_dir,
            download_main=args.download_main,
            download_workshops=args.download_workshops,
            download_supplement=args.download_supplement,
            time_step=args.time_step,
            downloader=None if args.downloader == 'requests' else 'IDM'
        )
        
        if success:
            print("🎉 Debug completed successfully!")
        else:
            print("💥 Debug failed!")
            sys.exit(1)

if __name__ == '__main__':
    main()
