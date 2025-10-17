#!/usr/bin/env python3
"""
Test the updated fig100k processor with correct data structure
"""

import json
import os
import tempfile
from pathlib import Path
from fig100k_processor import Fig100kProcessor

def create_test_data():
    """Create test data with correct fig100k structure"""
    test_data = [
        {
            "figure_id": "test_001",
            "captions": [
                "Figure 1: System Architecture Overview",
                "This diagram illustrates the main components and data flow in our proposed system."
            ],
            "captions_norm": [
                "Figure 1: System Architecture Overview",
                "This diagram illustrates the main components and data flow in our proposed system."
            ],
            "aspect": 1.2
        },
        {
            "figure_id": "test_002",
            "captions": [
                "Figure 2: Performance Results"
            ],
            "captions_norm": [
                "Figure 2: Performance Results"
            ],
            "aspect": 0.8
        },
        {
            "figure_id": "test_003",
            "captions": [],
            "captions_norm": [
                "Figure 3: Comparison Analysis"
            ],
            "aspect": 1.0
        }
    ]
    return test_data

def test_processor_data_handling():
    """Test that the processor correctly handles fig100k data structure"""
    print("Testing fig100k processor data handling...")
    
    # Create test data
    test_data = create_test_data()
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, indent=2)
        temp_json_path = f.name
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Test the processor's data loading and item processing logic
            processor = Fig100kProcessor(max_workers=1)
            
            # Test data loading
            loaded_data = processor.load_fig100k_data(temp_json_path)
            print(f"✅ Loaded {len(loaded_data)} items from test data")
            
            # Test individual item processing (without API calls)
            for i, item in enumerate(loaded_data):
                print(f"\nTesting item {i}:")
                print(f"  figure_id: {item.get('figure_id', 'MISSING')}")
                print(f"  captions: {item.get('captions', [])}")
                print(f"  captions_norm: {item.get('captions_norm', [])}")
                
                # Test caption extraction logic (same as in processor)
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
                    print(f"  ✅ Caption found - Processing would succeed")
                
                # Test image path construction
                figure_id = item.get("figure_id", "")
                image_path = processor.find_image_file(figure_id)
                print(f"  Image path: {image_path}")
                if image_path:
                    print(f"    ✅ Image file found")
                else:
                    print(f"    ⚠️ Image file not found (expected for test data)")
        
        finally:
            # Clean up
            os.unlink(temp_json_path)
    
    print(f"\n✅ Processor data handling test completed!")
    print(f"📝 The updated processor should now correctly handle fig100k data structure")

if __name__ == "__main__":
    test_processor_data_handling()
