#!/usr/bin/env python3
"""
Test script to verify image file finding works with actual fig100k data
"""

import json
import os
from typing import List, Dict, Any

def test_image_finding():
    """Test image file finding with actual figure_ids from the JSON data"""
    
    # Create a mock processor (we only need the find_image_file method)
    class MockProcessor:
        def find_image_file(self, figure_id: str, possible_dirs: List[str] = None) -> str:
            """Find the actual image file for a figure_id"""
            if possible_dirs is None:
                possible_dirs = [
                    "Paper2Fig100k",
                    "/blob/yasheng/Paper2Fig100k",
                    "/dev/shm/yasheng/Paper2Fig100k",
                    "images",
                    "data"
                ]
            
            image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
            
            for base_dir in possible_dirs:
                for ext in image_extensions:
                    image_path = os.path.join(base_dir, f"{figure_id}{ext}")
                    if os.path.exists(image_path):
                        return image_path
            
            return ""
    
    processor = MockProcessor()
    
    # Test with figure_ids from the JSON snippet you provided
    test_figure_ids = [
        "2010.16417v1-Figure2-1",
        "1709.00770v1-Figure4-1", 
        "2112.09854v1-Figure8-1",
        "1710.10393v1-Figure1-1"
    ]
    
    print("Testing image file finding with actual figure_ids...")
    print("=" * 60)
    
    found_count = 0
    for figure_id in test_figure_ids:
        image_path = processor.find_image_file(figure_id)
        if image_path:
            print(f"✅ {figure_id} -> {image_path}")
            found_count += 1
        else:
            print(f"❌ {figure_id} -> Not found")
    
    print("=" * 60)
    print(f"Found {found_count}/{len(test_figure_ids)} images")
    
    if found_count == len(test_figure_ids):
        print("🎉 All test images found! The find_image_file method works correctly.")
        return True
    else:
        print("⚠️ Some images not found. Check the directory paths.")
        return False

def test_with_json_sample():
    """Test with a small sample from the actual JSON file"""
    print("\nTesting with JSON sample data...")
    print("=" * 60)
    
    # Sample data from the JSON snippet you provided
    sample_data = [
        {
            "figure_id": "2010.16417v1-Figure2-1",
            "captions": ["Figure 2: The overall pipeline of MichiGAN..."],
            "captions_norm": ["figure number-tk the overall pipeline of michigan..."],
            "aspect": 1.38
        },
        {
            "figure_id": "1709.00770v1-Figure4-1", 
            "captions": ["Figure 4: Overall input and output of our framework"],
            "captions_norm": ["figure number-tk overall input and output of our framework"],
            "aspect": 2.28
        }
    ]
    
    class MockProcessor:
        def find_image_file(self, figure_id: str, possible_dirs: List[str] = None) -> str:
            if possible_dirs is None:
                possible_dirs = [
                    "Paper2Fig100k",
                    "/blob/yasheng/Paper2Fig100k",
                    "/dev/shm/yasheng/Paper2Fig100k",
                    "images",
                    "data"
                ]
            
            image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
            
            for base_dir in possible_dirs:
                for ext in image_extensions:
                    image_path = os.path.join(base_dir, f"{figure_id}{ext}")
                    if os.path.exists(image_path):
                        return image_path
            return ""
    
    processor = MockProcessor()
    
    for i, item in enumerate(sample_data):
        figure_id = item["figure_id"]
        caption = item["captions"][0] if item["captions"] else "No caption"
        
        print(f"\nItem {i+1}:")
        print(f"  figure_id: {figure_id}")
        print(f"  caption: {caption[:50]}...")
        
        image_path = processor.find_image_file(figure_id)
        if image_path:
            print(f"  ✅ Image found: {image_path}")
        else:
            print(f"  ❌ Image not found")

if __name__ == "__main__":
    success = test_image_finding()
    test_with_json_sample()
    
    if success:
        print("\n🎉 Image finding test completed successfully!")
        print("The fig100k_processor.py should now be able to find the actual image files.")
    else:
        print("\n⚠️ Some issues found. Please check the configuration.")
