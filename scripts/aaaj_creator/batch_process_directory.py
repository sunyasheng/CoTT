#!/usr/bin/env python3
"""
批量处理目录中的所有图片，生成 DPG 标注
支持过滤 reconstructed 图片，只处理原图

Usage:
    python batch_process_directory.py --input-dir /blob/yasheng/Paper2Fig100k_flux_train --output-dir /path/to/output --api-type azure
"""

import argparse
import os
import sys
from pathlib import Path
from diagram_parse_graph_creator import DiagramParseGraphCreator
import json
from typing import List, Dict
from tqdm import tqdm
import time

def find_original_images(input_dir: str, max_images: int = None) -> List[Path]:
    """查找所有原图（排除 _reconstructed 图片）"""
    input_path = Path(input_dir)
    images = []
    
    print(f"🔍 Searching for original images in: {input_dir}")
    
    # 查找所有 .png 文件
    for png_file in input_path.rglob("*.png"):
        # 排除 _reconstructed 图片
        if "_reconstructed" not in png_file.name.lower():
            images.append(png_file)
            
            # 如果指定了最大数量，限制结果
            if max_images and len(images) >= max_images:
                break
    
    return sorted(images)

def process_images_batch(images: List[Path], creator: DiagramParseGraphCreator, 
                         output_base_dir: str, batch_size: int = 100,
                         start_idx: int = 0, max_images: int = None) -> Dict:
    """批量处理图片，支持分批处理"""
    
    if max_images:
        images = images[:max_images]
    
    total = len(images)
    results = {
        "total": total,
        "success": 0,
        "failed": 0,
        "details": []
    }
    
    print(f"📊 Processing {total} images (batch size: {batch_size})")
    print(f"Starting from index: {start_idx}")
    
    # 创建输出目录
    output_path = Path(output_base_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 使用 tqdm 显示进度
    for i in tqdm(range(start_idx, total), desc="Processing images", initial=start_idx, total=total):
        image_path = images[i]
        
        try:
            # 确定输出路径：保持相对目录结构
            relative_path = image_path.relative_to(image_path.parents[len(image_path.parents)-1])
            output_file = output_path / relative_path.parent / f"{image_path.stem}_dpg.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 跳过已存在的文件
            if output_file.exists():
                print(f"\n⏭️  Skipping {image_path.name} (already exists)")
                results["success"] += 1
                results["details"].append({
                    "image": str(image_path),
                    "output": str(output_file),
                    "status": "skipped",
                    "blobs": 0,
                    "text_boxes": 0,
                    "arrows": 0
                })
                continue
            
            # 处理图片
            result = creator.analyze_diagram(str(image_path), str(output_file))
            
            # 提取统计信息
            dpg = result.get("dpg", {})
            constituents = dpg.get("constituents", {})
            relationships = dpg.get("relationships", {})
            
            stats = {
                "image": str(image_path),
                "output": str(output_file),
                "status": "success",
                "blobs": len(constituents.get("blobs", [])),
                "text_boxes": len(constituents.get("text_boxes", [])),
                "arrows": len(constituents.get("arrows", [])),
                "arrow_heads": len(constituents.get("arrow_heads", [])),
                "relationships": {k: len(v) for k, v in relationships.items() if v}
            }
            
            results["success"] += 1
            results["details"].append(stats)
            
            # 每处理一定数量后保存中间结果
            if (i + 1) % batch_size == 0:
                summary_file = output_path / f"batch_summary_checkpoint_{i+1}.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Checkpoint saved at {i+1} images")
            
        except Exception as e:
            print(f"\n❌ Failed {image_path.name}: {e}")
            results["failed"] += 1
            results["details"].append({
                "image": str(image_path),
                "status": "failed",
                "error": str(e)
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Batch process all images in a directory to create DPG annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in directory
  python batch_process_directory.py --input-dir /blob/yasheng/Paper2Fig100k_flux_train --output-dir /path/to/output --api-type azure
  
  # Process first 1000 images
  python batch_process_directory.py --input-dir /blob/yasheng/Paper2Fig100k_flux_train --output-dir /path/to/output --api-type azure --max-images 1000
  
  # Resume from image 5000
  python batch_process_directory.py --input-dir /blob/yasheng/Paper2Fig100k_flux_train --output-dir /path/to/output --api-type azure --start-idx 5000
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory containing diagram images'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for DPG JSON files'
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
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process (for testing)'
    )
    
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Start processing from this image index (for resuming)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Save checkpoint every N images (default: 100)'
    )
    
    parser.add_argument(
        '--summary-output',
        type=str,
        default=None,
        help='Path to save final summary JSON file'
    )
    
    args = parser.parse_args()
    
    # 查找所有原图
    images = find_original_images(args.input_dir, args.max_images)
    
    if not images:
        print("❌ No original images found!")
        sys.exit(1)
    
    print(f"✅ Found {len(images)} original images")
    if args.start_idx > 0:
        images = images[args.start_idx:]
        print(f"📌 Starting from index {args.start_idx}, {len(images)} images remaining")
    
    # 创建 DPG creator
    try:
        creator = DiagramParseGraphCreator(
            api_key=args.api_key,
            api_base=args.api_base,
            api_type=args.api_type
        )
        print(f"✅ DPG creator initialized (API type: {args.api_type})")
    except Exception as e:
        print(f"❌ Error initializing DPG creator: {e}")
        sys.exit(1)
    
    # 批量处理
    print(f"\n🚀 Starting batch processing...")
    start_time = time.time()
    
    results = process_images_batch(
        images, 
        creator, 
        args.output_dir,
        batch_size=args.batch_size,
        start_idx=0,  # images already sliced
        max_images=args.max_images
    )
    
    elapsed_time = time.time() - start_time
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"📊 Batch Processing Summary")
    print(f"{'='*80}")
    print(f"Total images: {results['total']}")
    print(f"✅ Success: {results['success']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"⏱️  Time elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    if results['success'] > 0:
        print(f"⚡ Average time per image: {elapsed_time/results['success']:.2f} seconds")
    
    if results['success'] > 0:
        print(f"\n📈 Average statistics (successful):")
        successful_details = [d for d in results["details"] if d.get("status") == "success"]
        if successful_details:
            avg_stats = {
                "blobs": sum(d.get("blobs", 0) for d in successful_details) / len(successful_details),
                "text_boxes": sum(d.get("text_boxes", 0) for d in successful_details) / len(successful_details),
                "arrows": sum(d.get("arrows", 0) for d in successful_details) / len(successful_details),
            }
            print(f"  - Blobs: {avg_stats['blobs']:.1f}")
            print(f"  - Text Boxes: {avg_stats['text_boxes']:.1f}")
            print(f"  - Arrows: {avg_stats['arrows']:.1f}")
    
    # 保存总结
    if args.summary_output:
        summary_path = args.summary_output
    else:
        summary_path = os.path.join(args.output_dir, "batch_dpg_summary.json")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()

