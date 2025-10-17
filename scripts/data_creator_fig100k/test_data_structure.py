#!/usr/bin/env python3
"""
Test script to verify the fig100k data structure and processor compatibility
"""

import json
import os
from pathlib import Path

def test_fig100k_data_structure():
    """Test the fig100k data structure"""
    
    # Create sample fig100k data structure
    sample_data = [
        {
            "figure_id": "test_figure_001",
            "captions": [
                "Figure 1: Overview of the proposed architecture",
                "This figure shows the main components of our system including the input layer, processing units, and output generation."
            ],
            "captions_norm": [
                "Figure 1: Overview of the proposed architecture",
                "This figure shows the main components of our system including the input layer, processing units, and output generation."
            ],
            "aspect": 1.5
        },
        {
            "figure_id": "test_figure_002", 
            "captions": [
                "Figure 2: Performance comparison"
            ],
            "captions_norm": [
                "Figure 2: Performance comparison"
            ],
            "aspect": 0.8
        },
        {
            "figure_id": "test_figure_003",
            "captions": [],
            "captions_norm": [
                "Figure 3: Results analysis"
            ],
            "aspect": 1.2
        }
    ]
    
    # Test data structure validation
    print("Testing fig100k data structure...")
    
    for i, item in enumerate(sample_data):
        print(f"\nItem {i}:")
        print(f"  figure_id: {item.get('figure_id', 'MISSING')}")
        print(f"  captions: {item.get('captions', [])}")
        print(f"  captions_norm: {item.get('captions_norm', [])}")
        print(f"  aspect: {item.get('aspect', 'MISSING')}")
        
        # Test caption extraction logic
        captions = item.get("captions", [])
        captions_norm = item.get("captions_norm", [])
        
        if captions and len(captions) > 0:
            caption = captions[0]
            context = "; ".join(captions[1:]) if len(captions) > 1 else ""
        elif captions_norm and len(captions_norm) > 0:
            caption = captions_norm[0]
            context = "; ".join(captions_norm[1:]) if len(captions_norm) > 1 else ""
        else:
            caption = ""
            context = ""
        
        print(f"  Extracted caption: '{caption}'")
        print(f"  Extracted context: '{context}'")
        
        if not caption:
            print(f"  ❌ MISSING CAPTION - This would cause processing to fail")
        else:
            print(f"  ✅ Caption found")
    
    # Test image path construction
    print(f"\nTesting image path construction...")
    figure_id = "test_figure_001"
    possible_paths = [
        f"Paper2Fig100k/{figure_id}.png",
        f"/blob/yasheng/Paper2Fig100k/{figure_id}.png",
        f"/dev/shm/yasheng/Paper2Fig100k/{figure_id}.png"
    ]
    
    for path in possible_paths:
        print(f"  Possible path: {path}")
        if os.path.exists(path):
            print(f"    ✅ File exists")
        else:
            print(f"    ❌ File not found")
    
    print(f"\n✅ Data structure test completed!")
    print(f"📝 Summary:")
    print(f"  - fig100k uses 'captions' (array) not 'caption' (string)")
    print(f"  - fig100k uses 'figure_id' to construct image paths")
    print(f"  - Image paths are constructed as 'Paper2Fig100k/{figure_id}.png'")
    print(f"  - Both 'captions' and 'captions_norm' arrays are available")

if __name__ == "__main__":
    test_fig100k_data_structure()
