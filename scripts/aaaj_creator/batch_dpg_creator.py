#!/usr/bin/env python3
"""
Batch Diagram Parse Graph Creator
批量处理 paper_pics 文件夹中的所有原图，生成 DPG 标注

Usage:
    python batch_dpg_creator.py [--base-dir BASE_DIR] [--api-type azure|openai]
"""

import argparse
import os
import sys
from pathlib import Path
import json
from typing import List, Dict

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagram_parse_graph_creator import DiagramParseGraphCreator

def find_original_images(base_dir: str) -> List[Path]:
    """查找所有原图（排除 _reconstructed 图片）"""
    base_path = Path(base_dir)
    images = []
    
    # 查找所有子文件夹中的 .png 文件
    for png_file in base_path.rglob("*.png"):
        # 排除 _reconstructed 图片
        if "_reconstructed" not in png_file.name:
            images.append(png_file)
    
    return sorted(images)

def process_images(images: List[Path], creator: DiagramParseGraphCreator, output_base_dir: str = None) -> Dict:
    """批量处理图片"""
    results = {
        "total": len(images),
        "success": 0,
        "failed": 0,
        "details": []
    }
    
    for i, image_path in enumerate(images, 1):
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(images)}: {image_path.name}")
        print(f"{'='*80}")
        
        try:
            # 确定输出路径
            if output_base_dir:
                output_dir = Path(output_base_dir) / image_path.parent.name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}_dpg.json"
            else:
                output_path = image_path.parent / f"{image_path.stem}_dpg.json"
            
            # 处理图片
            result = creator.analyze_diagram(str(image_path), str(output_path))
            
            # 提取统计信息
            dpg = result.get("dpg", {})
            constituents = dpg.get("constituents", {})
            relationships = dpg.get("relationships", {})
            
            stats = {
                "image": str(image_path),
                "output": str(output_path),
                "status": "success",
                "blobs": len(constituents.get("blobs", [])),
                "text_boxes": len(constituents.get("text_boxes", [])),
                "arrows": len(constituents.get("arrows", [])),
                "arrow_heads": len(constituents.get("arrow_heads", [])),
                "relationships": {k: len(v) for k, v in relationships.items() if v}
            }
            
            results["success"] += 1
            results["details"].append(stats)
            
            print(f"✅ Success: {stats['blobs']} blobs, {stats['text_boxes']} text boxes, {stats['arrows']} arrows")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results["failed"] += 1
            results["details"].append({
                "image": str(image_path),
                "status": "failed",
                "error": str(e)
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Batch process diagrams to create DPG annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/Users/suny0a/Proj/MM-Reasoning/IMAGEGEN/DataPrep(CoTT)/debug/paper_pics',
        help='Base directory containing diagram images'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for DPG JSON files (default: same as image directory)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key'
    )
    
    parser.add_argument(
        '--api-base',
        type=str,
        default=None,
        help='API base URL for Azure OpenAI'
    )
    
    parser.add_argument(
        '--api-type',
        type=str,
        choices=['openai', 'azure'],
        default='azure',
        help='API type: openai or azure (default: azure)'
    )
    
    parser.add_argument(
        '--summary-output',
        type=str,
        default=None,
        help='Path to save summary JSON file'
    )
    
    args = parser.parse_args()
    
    # 查找所有原图
    print(f"🔍 Searching for original images in: {args.base_dir}")
    images = find_original_images(args.base_dir)
    
    if not images:
        print("❌ No original images found!")
        sys.exit(1)
    
    print(f"✅ Found {len(images)} original images:")
    for img in images:
        print(f"   - {img}")
    
    # 创建 DPG creator
    try:
        creator = DiagramParseGraphCreator(
            api_key=args.api_key,
            api_base=args.api_base,
            api_type=args.api_type
        )
    except Exception as e:
        print(f"❌ Error initializing DPG creator: {e}")
        sys.exit(1)
    
    # 批量处理
    print(f"\n🚀 Starting batch processing...")
    results = process_images(images, creator, args.output_dir)
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"📊 Batch Processing Summary")
    print(f"{'='*80}")
    print(f"Total images: {results['total']}")
    print(f"✅ Success: {results['success']}")
    print(f"❌ Failed: {results['failed']}")
    
    if results['success'] > 0:
        print(f"\n📈 Average statistics (successful):")
        avg_stats = {
            "blobs": sum(d.get("blobs", 0) for d in results["details"] if d.get("status") == "success") / results["success"],
            "text_boxes": sum(d.get("text_boxes", 0) for d in results["details"] if d.get("status") == "success") / results["success"],
            "arrows": sum(d.get("arrows", 0) for d in results["details"] if d.get("status") == "success") / results["success"],
        }
        print(f"  - Blobs: {avg_stats['blobs']:.1f}")
        print(f"  - Text Boxes: {avg_stats['text_boxes']:.1f}")
        print(f"  - Arrows: {avg_stats['arrows']:.1f}")
    
    # 保存总结
    if args.summary_output:
        with open(args.summary_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Summary saved to: {args.summary_output}")
    
    # 打印详细结果
    print(f"\n📋 Detailed Results:")
    for detail in results["details"]:
        if detail.get("status") == "success":
            print(f"  ✅ {Path(detail['image']).name}: {detail['blobs']} blobs, {detail['text_boxes']} text, {detail['arrows']} arrows")
        else:
            print(f"  ❌ {Path(detail['image']).name}: {detail.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()

