#!/usr/bin/env python3
"""
CVF paper download example script
Supports downloading papers from CVPR, ICCV, WACV, ACCV conferences
"""

import os
import sys

# Add lib directory to Python path
root_folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)

from code.paper_downloader_CVF import download_paper

def download_cvpr_papers():
    """Download CVPR 2024 papers example"""
    print("Starting CVPR 2024 download...")
    
    download_paper(
        year=2024,
        conference='CVPR',
        save_dir='/Users/suny0a/Proj/CoTT/papers/CVPR',
        is_download_main_paper=True,      # Download main papers
        is_download_supplement=True,      # Download supplements
        time_step_in_seconds=5,          # Download interval 5 seconds
        is_download_main_conference=True, # Download main conference papers
        is_download_workshops=False,      # Skip workshops for now
        downloader=None,                  # Use Python requests
    )
    print("CVPR 2024 download completed!")

def download_iccv_papers():
    """Download ICCV 2023 papers example"""
    print("Starting ICCV 2023 download...")
    
    download_paper(
        year=2023,
        conference='ICCV',
        save_dir='/Users/suny0a/Proj/CoTT/papers/ICCV',
        is_download_main_paper=True,
        is_download_supplement=True,
        time_step_in_seconds=5,
        is_download_main_conference=True,
        is_download_workshops=False,
        downloader=None,
    )
    print("ICCV 2023 download completed!")

def download_wacv_papers():
    """Download WACV 2024 papers example"""
    print("Starting WACV 2024 download...")
    
    download_paper(
        year=2024,
        conference='WACV',
        save_dir='/Users/suny0a/Proj/CoTT/papers/WACV',
        is_download_main_paper=True,
        is_download_supplement=True,
        time_step_in_seconds=5,
        is_download_main_conference=True,
        is_download_workshops=False,
        downloader=None,
    )
    print("WACV 2024 download completed!")

if __name__ == '__main__':
    print("CVF Paper Downloader Example")
    print("=" * 50)
    
    # Create save directory
    base_dir = '/Users/suny0a/Proj/CoTT/papers'
    os.makedirs(base_dir, exist_ok=True)
    
    # Choose conference to download
    print("Please choose conference to download:")
    print("1. CVPR 2024")
    print("2. ICCV 2023") 
    print("3. WACV 2024")
    print("4. Download all")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == '1':
        download_cvpr_papers()
    elif choice == '2':
        download_iccv_papers()
    elif choice == '3':
        download_wacv_papers()
    elif choice == '4':
        download_cvpr_papers()
        download_iccv_papers()
        download_wacv_papers()
    else:
        print("Invalid choice, exiting program")
