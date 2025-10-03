#!/usr/bin/env python3
"""
Universal paper analyzer for any markdown file.
This script can analyze any academic paper markdown file and categorize its images.
"""

import os
import sys
import argparse
from pathlib import Path
from markdown_image_analyzer import MarkdownImageAnalyzer


def analyze_paper(markdown_path: str, images_dir: str = None):
    """Analyze any paper markdown file."""
    
    # Check if files exist
    if not Path(markdown_path).exists():
        print(f"Error: Markdown file not found: {markdown_path}")
        return
    
    if images_dir and not Path(images_dir).exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    # Initialize analyzer
    analyzer = MarkdownImageAnalyzer(markdown_path, images_dir)
    
    # Read markdown content
    print("Reading markdown content...")
    content = analyzer.read_markdown()
    
    if not content:
        print("Failed to read markdown file")
        return
    
    print(f"Content length: {len(content)} characters")
    print()
    
    # Extract images
    print("Extracting images...")
    images = analyzer.extract_images()
    print(f"Found {len(images)} images")
    
    # Show all images
    print("\nAll images found:")
    for i, img in enumerate(images, 1):
        status = "✓" if img['exists'] else "✗"
        print(f"  {i:2d}. {status} {img['filename']}")
        if img['alt_text']:
            print(f"      Alt text: {img['alt_text']}")
        print(f"      Path: {img['path']}")
        print()
    
    # Categorize images
    print("Categorizing images...")
    categories = analyzer.categorize_images()
    
    print("\n=== Image Categories ===")
    for category, imgs in categories.items():
        if imgs:
            print(f"\n{category.upper().replace('_', ' ')} ({len(imgs)} images):")
            for img in imgs:
                print(f"  - {img['filename']}")
                if img['alt_text']:
                    print(f"    Alt: {img['alt_text']}")
                
                # Show context around the image
                context = analyzer.get_context_around_image(img['path'], context_lines=3)
                if context:
                    print(f"    Context: {context[:100]}...")
                print()
    
    # Generate summary
    print("=== Summary ===")
    summary = analyzer.generate_summary()
    print(f"Total images: {summary['total_images']}")
    print(f"Existing images: {summary['existing_images']}")
    print()
    
    for category, info in summary['categories'].items():
        if info['count'] > 0:
            print(f"{category}: {info['count']} images")
    
    # Save results
    output_path = analyzer.save_results()
    print(f"\nDetailed results saved to: {output_path}")
    
    # Show specific categories of interest
    print("\n=== Key Image Types ===")
    key_categories = ['teaser', 'qualitative', 'workflow_framework']
    
    for category in key_categories:
        if category in categories and categories[category]:
            print(f"\n{category.upper().replace('_', ' ')} Images:")
            for img in categories[category]:
                print(f"  - {img['filename']}: {img['alt_text']}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Analyze images in academic paper markdown files')
    parser.add_argument('markdown_path', help='Path to the markdown file')
    parser.add_argument('--images-dir', help='Path to the images directory (optional, will be auto-detected)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    analyze_paper(args.markdown_path, args.images_dir)


if __name__ == "__main__":
    main()
