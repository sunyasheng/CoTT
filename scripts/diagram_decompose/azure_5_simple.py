#!/usr/bin/env python3
"""
Simplified diagram decomposition with GPT-5 (no GroundingDINO dependency).

This script uses GPT-5 Vision to:
1. Analyze scientific figures and identify visual elements
2. Generate captions for elements that need to be cropped
3. Provide detailed reconstruction instructions
4. Export bounding boxes and crop information

Features:
- No GroundingDINO dependency - just outputs captions for manual cropping
- Generates crop captions for material extraction
- Provides step-by-step reconstruction instructions
- Supports both detection and reconstruction workflows

Usage:
  python azure_5_simple.py \
    --image vis.png \
    --azure_endpoint https://<resource>.cognitiveservices.azure.com/ \
    --azure_api_key $AZURE_OPENAI_API_KEY \
    --deployment gpt-5 \
    --api_version 2025-01-01-preview \
    --out_json vis_analysis.json \
    --out_vis vis_analysis.png \
    --crop_dir crops/
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

# 尝试导入 dotenv，如果没有则使用简单的环境变量加载
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    print("⚠️ python-dotenv 未安装，使用简单环境变量加载")

def load_env_vars():
    """加载环境变量"""
    # 尝试从多个位置加载 .env 文件
    env_paths = [
        Path(__file__).parent / ".env",  # 当前目录
        Path(__file__).parent.parent / ".env",  # 上级目录
        Path(__file__).parent.parent.parent / ".env",  # 根目录
        Path(__file__).parent.parent.parent / ".env_old",  # 根目录的 .env_old
    ]
    
    if HAS_DOTENV:
        for env_path in env_paths:
            if env_path.exists():
                print(f"📄 加载环境变量: {env_path}")
                load_dotenv(env_path)
                return True
    else:
        # 简单的环境变量加载（从 .env_old 文件）
        env_old_path = Path(__file__).parent.parent.parent / ".env_old"
        if env_old_path.exists():
            print(f"📄 从 .env_old 加载环境变量: {env_old_path}")
            with open(env_old_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
                
                # 设置默认的 Azure OpenAI 配置（如果没有设置的话）
                if not os.getenv("AZURE_OPENAI_ENDPOINT"):
                    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com/"
                if not os.getenv("AZURE_OPENAI_DEPLOYMENT"):
                    os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-5"
                if not os.getenv("AZURE_OPENAI_API_VERSION"):
                    os.environ["AZURE_OPENAI_API_VERSION"] = "2025-01-01-preview"
                print(f"   ✅ 设置默认 Azure OpenAI 配置")
            return True
    
    print("⚠️ 未找到环境变量文件")
    return False

# 加载环境变量
load_env_vars()


def _b64_image(path: Path, max_size: int = 1024) -> tuple:
    """Load image, scale if too large, return (base64_data, actual_size, scale_factor)."""
    img = Image.open(path)
    original_size = img.size
    
    # Only scale down if image is too large, maintain aspect ratio
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_w = int(original_size[0] * scale)
        new_h = int(original_size[1] * scale)
        img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        actual_size = (new_w, new_h)
        scale_info = (scale, 0, 0)  # No padding offset
    else:
        img_scaled = img
        actual_size = original_size
        scale_info = (1.0, 0, 0)  # No scaling, no padding
    
    # Save to bytes
    import io
    buffer = io.BytesIO()
    img_scaled.save(buffer, format='PNG', optimize=True)
    data = buffer.getvalue()
    
    print(f"Image size: {actual_size[0]}x{actual_size[1]} (original: {original_size[0]}x{original_size[1]})")
    print(f"Scale factor: {scale_info[0]:.3f}, No padding")
    
    return base64.b64encode(data).decode("utf-8"), actual_size, scale_info


def analyze_with_gpt5(image_path: Path, cfg: Dict) -> Dict:
    """Analyze image using GPT-5 Vision."""
    
    # Create the image for GPT-5 analysis (no padding)
    image_b64, image_size, scale_info = _b64_image(image_path, max_size=1024)
    width, height = image_size
    scale, x_offset, y_offset = scale_info
    
    # Build endpoint URL
    url = (
        f"{cfg['endpoint']}openai/deployments/{cfg['deployment']}/chat/completions?"
        f"api-version={cfg['api_version']}"
    )
    headers = {"Content-Type": "application/json", "api-key": cfg["api_key"]}
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"""You are analyzing a scientific figure to understand its structure and create reconstruction instructions.

This image is {width}x{height} pixels.

Your task is to:
1. Identify all visual elements (panels, labels, arrows, legends, etc.)
2. Understand the layout and relationships between elements
3. Generate crop captions for elements that need to be extracted as materials
4. Provide detailed reconstruction instructions

Please provide your analysis in the following JSON format:
{{
  "analysis": {{
    "figure_type": "description of the figure type",
    "layout_structure": "description of overall layout",
    "main_components": ["list of main visual components"],
    "relationships": "how components relate to each other"
  }},
  "elements": [
    {{
      "id": "element_1",
      "type": "panel|label|arrow|legend|chart|image|other",
      "description": "detailed description of this element",
      "bbox": [x_min, y_min, x_max, y_max],
      "needs_crop": true/false,
      "crop_caption": "caption for GroundingDINO to extract this element (if needs_crop=true)",
      "importance": "high|medium|low"
    }}
  ],
  "reconstruction_instructions": {{
    "overall_approach": "general strategy for recreating this figure",
    "step_by_step": [
      "Step 1: description",
      "Step 2: description",
      "..."
    ],
    "required_materials": [
      {{
        "element_id": "element_1",
        "caption": "caption for cropping",
        "usage": "how this material will be used in reconstruction"
      }}
    ],
    "layout_guidelines": "specific instructions for positioning and sizing",
    "styling_notes": "color schemes, fonts, line styles, etc."
  }}
}}

IMPORTANT: 
- Use pixel coordinates (0 to {width-1} for x, 0 to {height-1} for y)
- Be specific in crop captions for GroundingDINO
- Provide clear, actionable reconstruction steps
- Focus on elements that are essential for recreating the figure"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"},
                    },
                ],
            },
        ],
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        
        print(f"GPT-5 response length: {len(content)} characters")
        
        # Try to parse as JSON
        try:
            result = json.loads(content)
            result["scale_info"] = scale_info
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["scale_info"] = scale_info
                return result
            else:
                return {
                    "error": f"no_json_found_in_response: {content[:500]}...",
                    "raw_response": content,
                    "scale_info": scale_info
                }
                
    except Exception as exc:
        return {
            "error": f"gpt5_analysis_failed: {exc}",
            "scale_info": scale_info
        }


def convert_coordinates_to_original(elements: List[Dict], scale_info: Tuple, original_size: Tuple) -> List[Dict]:
    """Convert coordinates from padded image back to original image coordinates."""
    if not scale_info or not elements:
        return elements
    
    scale, x_offset, y_offset = scale_info
    orig_w, orig_h = original_size
    
    converted_elements = []
    for element in elements:
        if "bbox" in element and len(element["bbox"]) == 4:
            x_min, y_min, x_max, y_max = element["bbox"]
            
            # Remove padding offset and scale back to original size
            x_min_orig = (x_min - x_offset) / scale
            y_min_orig = (y_min - y_offset) / scale
            x_max_orig = (x_max - x_offset) / scale
            y_max_orig = (y_max - y_offset) / scale
            
            # Clamp to original image bounds
            x_min_orig = max(0, min(x_min_orig, orig_w))
            y_min_orig = max(0, min(y_min_orig, orig_h))
            x_max_orig = max(0, min(x_max_orig, orig_w))
            y_max_orig = max(0, min(y_max_orig, orig_h))
            
            # Ensure valid bounding box
            if x_max_orig <= x_min_orig:
                x_max_orig = x_min_orig + 1
            if y_max_orig <= y_min_orig:
                y_max_orig = y_min_orig + 1
            
            # Convert to normalized coordinates
            x_norm = x_min_orig / orig_w
            y_norm = y_min_orig / orig_h
            w_norm = (x_max_orig - x_min_orig) / orig_w
            h_norm = (y_max_orig - y_min_orig) / orig_h
            
            element_copy = element.copy()
            element_copy["bbox_original"] = [int(x_min_orig), int(y_min_orig), int(x_max_orig), int(y_max_orig)]
            element_copy["bbox_norm"] = [x_norm, y_norm, w_norm, h_norm]
            converted_elements.append(element_copy)
        else:
            converted_elements.append(element)
    
    return converted_elements


def draw_analysis_overlay(image_path: Path, elements: List[Dict], out_path: Path, scale_info: Tuple = None) -> None:
    """Draw overlay showing detected elements and their types."""
    # Load the original image
    im_orig = Image.open(image_path).convert("RGB")
    
    # If we have scale info, scale the image but don't pad
    if scale_info:
        scale, x_offset, y_offset = scale_info
        
        # Scale the original image if needed
        if scale != 1.0:
            new_w = int(im_orig.size[0] * scale)
            new_h = int(im_orig.size[1] * scale)
            im = im_orig.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            im = im_orig
    else:
        im = im_orig
    
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size=max(12, im.width // 100))
    except Exception:
        try:
            font = ImageFont.truetype("Arial.ttf", size=max(12, im.width // 100))
        except Exception:
            font = ImageFont.load_default()

    # Color palette for different element types
    type_colors = {
        "panel": (230, 57, 70),      # Red
        "label": (29, 53, 87),       # Dark blue
        "arrow": (69, 123, 157),     # Light blue
        "legend": (53, 194, 109),    # Green
        "chart": (255, 183, 3),      # Yellow
        "image": (244, 162, 97),     # Orange
        "other": (128, 128, 128)     # Gray
    }
    
    for i, element in enumerate(elements):
        if "bbox" not in element or len(element["bbox"]) != 4:
            continue
            
        x_min, y_min, x_max, y_max = element["bbox"]
        element_type = element.get("type", "other")
        color = type_colors.get(element_type, type_colors["other"])
        
        # Draw bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=max(2, im.width // 400))
        
        # Draw label
        label = f"{element.get('id', f'el{i}')} ({element_type})"
        if element.get("needs_crop"):
            label += " [CROP]"
        
        # Calculate text size and position
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        pad = 2
        
        # Draw background for text
        text_x = x_min
        text_y = max(0, y_min - th - 2 * pad)
        draw.rectangle([text_x, text_y, text_x + tw + 2 * pad, text_y + th + 2 * pad], fill=color)
        draw.text((text_x + pad, text_y + pad), label, fill=(255, 255, 255), font=font)
    
    im.save(out_path)


def generate_crop_instructions(elements: List[Dict], out_dir: Path) -> List[Dict]:
    """Generate crop instructions for elements that need cropping."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    crop_instructions = []
    for i, element in enumerate(elements):
        if not element.get("needs_crop") or not element.get("crop_caption"):
            continue
        
        crop_instructions.append({
            "element_id": element.get("id", f"element_{i}"),
            "caption": element["crop_caption"],
            "bbox_original": element.get("bbox_original", []),
            "bbox_norm": element.get("bbox_norm", []),
            "description": element.get("description", ""),
            "importance": element.get("importance", "medium"),
            "usage": "See reconstruction_instructions for usage details"
        })
    
    # Save crop instructions to file
    instructions_file = out_dir / "crop_instructions.json"
    with open(instructions_file, 'w', encoding='utf-8') as f:
        json.dump(crop_instructions, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(crop_instructions)} crop instructions to: {instructions_file}")
    
    return crop_instructions


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplified diagram decomposition with GPT-5")
    parser.add_argument("--image", required=True, help="Path to input image (PNG/JPG)")
    parser.add_argument("--out_json", default=None, help="Output JSON path; default next to image")
    parser.add_argument("--out_vis", default=None, help="Output visualization image; default next to image")
    parser.add_argument("--crop_dir", default=None, help="If set, save crop instructions into this directory")
    parser.add_argument("--temperature", type=float, default=0.1)
    
    # Azure configuration
    parser.add_argument("--azure_endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT", "https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com/"))
    parser.add_argument("--azure_api_key", default=os.getenv("AZURE_OPENAI_API_KEY", ""))
    parser.add_argument("--deployment", default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5"))
    parser.add_argument("--api_version", default=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"))
    
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    out_json = Path(args.out_json) if args.out_json else image_path.parent / f"{image_path.stem}_analysis.json"
    out_vis = Path(args.out_vis) if args.out_vis else image_path.parent / f"{image_path.stem}_analysis.png"

    cfg = {
        "endpoint": args.azure_endpoint,
        "api_key": args.azure_api_key,
        "deployment": args.deployment,
        "api_version": args.api_version,
    }

    # Analyze the image
    print("Starting analysis with GPT-5...")
    result = analyze_with_gpt5(image_path, cfg)
    
    if isinstance(result, dict) and result.get("error"):
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Analysis failed. Details saved to {out_json}")
        return

    # Convert coordinates to original image space
    elements = result.get("elements", [])
    scale_info = result.get("scale_info")
    original_size = Image.open(image_path).size
    
    if scale_info:
        elements = convert_coordinates_to_original(elements, scale_info, original_size)
        print(f"Converted {len(elements)} elements to original coordinates")

    # Draw visualization
    try:
        draw_analysis_overlay(image_path, elements, out_vis, scale_info)
        print(f"Visualization saved to: {out_vis}")
    except Exception as exc:
        print(f"Warning: visualization failed: {exc}")

    # Generate crop instructions if requested
    crop_instructions = []
    if args.crop_dir:
        try:
            crop_instructions = generate_crop_instructions(elements, Path(args.crop_dir))
            print(f"Generated {len(crop_instructions)} crop instructions")
        except Exception as exc:
            print(f"Warning: crop instruction generation failed: {exc}")

    # Prepare final report
    report = {
        "image_path": str(image_path.resolve()),
        "analysis": result.get("analysis", {}),
        "elements": elements,
        "reconstruction_instructions": result.get("reconstruction_instructions", {}),
        "crop_instructions": crop_instructions,
        "scale_info": scale_info,
        "original_image_size": original_size,
    }

    # Save JSON report
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Analysis report saved to: {out_json}")

    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Figure type: {result.get('analysis', {}).get('figure_type', 'Unknown')}")
    print(f"Total elements detected: {len(elements)}")
    print(f"Elements needing crop: {sum(1 for e in elements if e.get('needs_crop'))}")
    print(f"Crop instructions generated: {len(crop_instructions)}")
    
    if result.get("reconstruction_instructions"):
        print(f"\nReconstruction approach: {result['reconstruction_instructions'].get('overall_approach', 'Not specified')}")
        print(f"Required materials: {len(result['reconstruction_instructions'].get('required_materials', []))}")
    
    # Print crop captions for easy reference
    if crop_instructions:
        print(f"\n=== CROP CAPTIONS FOR GROUNDINGDINO ===")
        for i, instruction in enumerate(crop_instructions, 1):
            print(f"{i}. {instruction['caption']}")


if __name__ == "__main__":
    main()
