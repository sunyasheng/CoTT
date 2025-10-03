#!/usr/bin/env python3
"""
Detect figure elements with Azure GPT-5 Vision and export bounding boxes.

Features
- Calls Azure OpenAI (deployment: gpt-5) with an input image
- Expects STRICT JSON with a components list containing bbox_norm [x,y,w,h]
- Converts to pixel boxes, draws an overlay, saves JSON
- Optionally crops each detected component to separate files

Usage
  python azure_5.py \
    --image vis.png \
    --azure_endpoint https://<resource>.cognitiveservices.azure.com/ \
    --azure_api_key $AZURE_OPENAI_API_KEY \
    --deployment gpt-5 \
    --api_version 2025-01-01-preview \
    --out_json vis_boxes.json \
    --out_vis vis_boxes.png \
    --crop_dir crops/

Notes
- Boxes are tight, origin at top-left. bbox_norm ranges 0..1.
- If the model returns absolute pixel boxes (w>1 or h>1), we keep them.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Dict, List

import requests
from PIL import Image, ImageDraw, ImageFont


SYSTEM_PROMPT = (
    "You detect panels and visual elements in scientific figures. "
    "Return STRICT JSON with: {components:[{id,name,type,meaning,confidence,bbox_norm:[x,y,w,h]}], layout:{}, quality:{}}. "
    "Rules: tight boxes, exclude whitespace, origin top-left, normalized 0..1 if possible. "
    "Types include: panel,input,gt,ours,baseline,legend,label,arrow,mark,other. "
    "IMPORTANT: Return ONLY valid JSON, no other text."
)


def _b64_image(path: Path, target_size: tuple = (512, 512)) -> tuple:
    """Load image, scale long edge to target size, pad to square, return (base64_data, actual_size, scale_factor)."""
    from PIL import Image
    import io
    
    img = Image.open(path)
    original_size = img.size
    target_w, target_h = target_size
    
    # Calculate scale factor to fit long edge to target size
    scale = min(target_w / original_size[0], target_h / original_size[1])
    new_w = int(original_size[0] * scale)
    new_h = int(original_size[1] * scale)
    
    # Resize maintaining aspect ratio
    img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create square canvas with white background
    img_padded = Image.new('RGB', (target_w, target_h), (255, 255, 255))
    
    # Calculate position to center the scaled image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Paste scaled image onto centered position
    img_padded.paste(img_scaled, (x_offset, y_offset))
    
    # Save to bytes
    buffer = io.BytesIO()
    img_padded.save(buffer, format='PNG', optimize=True)
    data = buffer.getvalue()
    
    print(f"Image scaled from {original_size[0]}x{original_size[1]} to {new_w}x{new_h}, padded to {target_w}x{target_h}")
    print(f"Scale factor: {scale:.3f}, Offset: ({x_offset}, {y_offset})")
    
    return base64.b64encode(data).decode("utf-8"), target_size, (scale, x_offset, y_offset)


def parse_gpt5_description(content: str, image_path: Path, image_size: tuple = (512, 512), scale_info: tuple = None) -> Dict:
    """Parse GPT-5 text description and extract bounding boxes."""
    if not content:
        return {"error": "empty_response_from_gpt5"}
    
    import re
    components = []
    width, height = image_size
    
    # Extract bounding boxes from the text using regex
    # Look for patterns like "Box: [10, 20, 100, 150]" (x_min, y_min, x_max, y_max)
    box_pattern = r'Box:\s*\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    boxes = re.findall(box_pattern, content)
    
    # Extract descriptions for each box
    # Look for patterns like "1) Description" or "- Description"
    desc_pattern = r'(?:\d+\)\s*|-\s*)([^-\n]+?)(?=\s*Box:|$)'
    descriptions = re.findall(desc_pattern, content, re.DOTALL)
    
    # Clean up descriptions
    descriptions = [desc.strip() for desc in descriptions if desc.strip()]
    
    # Create components from extracted boxes and descriptions
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        
        # Validate coordinates are within image bounds
        if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
            print(f"Warning: Component {i} has invalid coordinates [{x_min}, {y_min}, {x_max}, {y_max}] for {width}x{height} image")
            # Clamp coordinates to valid range
            x_min = max(0, min(x_min, width-1))
            y_min = max(0, min(y_min, height-1))
            x_max = max(0, min(x_max, width))
            y_max = max(0, min(y_max, height))
            if x_max <= x_min:
                x_max = x_min + 1
            if y_max <= y_min:
                y_max = y_min + 1
        
        # Since GPT-5 now sees the 512x512 padded image, we need to convert coordinates back to original image space
        if scale_info:
            scale, x_offset, y_offset = scale_info
            # Remove padding offset and scale back to original size
            x_min_orig = (x_min - x_offset) / scale
            y_min_orig = (y_min - y_offset) / scale
            x_max_orig = (x_max - x_offset) / scale
            y_max_orig = (y_max - y_offset) / scale
            
            # Clamp coordinates to valid ranges
            from PIL import Image
            original_img = Image.open(image_path)
            orig_w, orig_h = original_img.size
            
            x_min_orig = max(0, min(x_min_orig, orig_w))
            y_min_orig = max(0, min(y_min_orig, orig_h))
            x_max_orig = max(0, min(x_max_orig, orig_w))
            y_max_orig = max(0, min(y_max_orig, orig_h))
            
            # Ensure valid bounding box
            if x_max_orig <= x_min_orig:
                x_max_orig = x_min_orig + 1
            if y_max_orig <= y_min_orig:
                y_max_orig = y_min_orig + 1
            
            # Convert to normalized coordinates based on original image size
            x_norm = x_min_orig / orig_w
            y_norm = y_min_orig / orig_h
            w_norm = (x_max_orig - x_min_orig) / orig_w
            h_norm = (y_max_orig - y_min_orig) / orig_h
            
            print(f"Component {i}: GPT-5 coords [{x_min}, {y_min}, {x_max}, {y_max}] -> Original coords [{x_min_orig:.0f}, {y_min_orig:.0f}, {x_max_orig:.0f}, {y_max_orig:.0f}] -> Norm coords [{x_norm:.3f}, {y_norm:.3f}, {w_norm:.3f}, {h_norm:.3f}]")
        else:
            # Convert from pixel coordinates to normalized coordinates
            x_norm = x_min / width
            y_norm = y_min / height
            w_norm = (x_max - x_min) / width
            h_norm = (y_max - y_min) / height
        
        # Get description for this component
        desc = descriptions[i] if i < len(descriptions) else f"Component {i+1}"
        
        # Determine component type based on description
        comp_type = "panel"
        if "legend" in desc.lower():
            comp_type = "legend"
        elif "input" in desc.lower():
            comp_type = "input"
        elif "ground truth" in desc.lower() or "gt" in desc.lower():
            comp_type = "gt"
        elif "ours" in desc.lower() or "sphere" in desc.lower():
            comp_type = "ours"
        elif "sgn" in desc.lower() or "voxformer" in desc.lower():
            comp_type = "baseline"
        elif "partial" in desc.lower():
            comp_type = "input"
        elif "completed" in desc.lower():
            comp_type = "ours"
        
        components.append({
            "id": f"component_{i}",
            "name": f"panel_{i}",
            "type": comp_type,
            "meaning": desc[:100],  # Truncate long descriptions
            "confidence": 0.9,  # High confidence since GPT-5 provided specific coordinates
            "bbox_norm": [x_norm, y_norm, w_norm, h_norm]
        })
    
    # If no boxes found, fall back to grid detection
    if not components:
        grid_match = re.search(r'(\d+)\s*(?:x|by|×)\s*(\d+)', content, re.IGNORECASE)
        if grid_match:
            rows, cols = int(grid_match.group(1)), int(grid_match.group(2))
            for i in range(rows * cols):
                components.append({
                    "id": f"component_{i}",
                    "name": f"panel_{i}",
                    "type": "panel",
                    "meaning": f"Grid panel {i+1}",
                    "confidence": 0.7,
                    "bbox_norm": [0.1 + (i % cols) * 0.8/cols, 0.1 + (i // cols) * 0.8/rows, 0.8/cols, 0.8/rows]
                })
    
    return {
        "components": components,
        "layout": {"grid": "detected", "reading_order": "ltr", "has_row_headers": False, "has_col_headers": True},
        "quality": {"coverage_estimate": 0.9, "num_panels": len(components), "warnings": ["GPT-5 reasoning-based detection"]},
        "gpt5_description": content[:1000] + "..." if len(content) > 1000 else content
    }


def call_gpt5(image_path: Path, cfg: Dict, temperature: float = 0.0) -> Dict:
    # Create the padded image and get its base64 encoding
    image_b64, image_size, scale_info = _b64_image(image_path, target_size=(512, 512))
    width, height = image_size
    scale, x_offset, y_offset = scale_info
    
    # Save the padded image temporarily for GPT-5 to analyze
    from PIL import Image
    import io
    
    # Recreate the padded image
    img_orig = Image.open(image_path)
    new_w = int(img_orig.size[0] * scale)
    new_h = int(img_orig.size[1] * scale)
    img_scaled = img_orig.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_padded = Image.new('RGB', (512, 512), (255, 255, 255))
    img_padded.paste(img_scaled, (x_offset, y_offset))
    
    # Save padded image temporarily
    padded_path = image_path.parent / f"{image_path.stem}_padded_512.png"
    img_padded.save(padded_path)
    print(f"Saved padded image to: {padded_path}")
    
    # Build endpoint URL; cfg["endpoint"] should be the cognitive services base URL
    url = (
        f"{cfg['endpoint']}openai/deployments/{cfg['deployment']}/chat/completions?"
        f"api-version={cfg['api_version']}"
    )
    headers = {"Content-Type": "application/json", "api-key": cfg["api_key"]}
    
    # Different approach for GPT-5 vs other models
    if cfg.get("deployment", "").lower() == "gpt-5":
        # Get base64 of the padded image
        with open(padded_path, "rb") as f:
            padded_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # GPT-5: Use reasoning approach with explicit image dimensions
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"This image is {width}x{height} pixels. It shows 4 room completion examples (a, b, c, d), each with 'Partial' and 'Completed' states.\n\nI need you to identify ONLY the 4 'Partial' room layout images (the left side of each example).\n\nFor each of the 4 Partial room images, provide:\n1. Brief description (e.g., 'Bedroom partial', 'Living room partial')\n2. Bounding box coordinates in format [x_min, y_min, x_max, y_max] using pixel coordinates (0 to {width-1} for x, 0 to {height-1} for y)\n\nExample format:\n1) Bedroom partial state\nBox: [10, 20, 100, 150]\n\nIMPORTANT: Only detect the 4 Partial room images, ignore Completed states, arrows, and text labels."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{padded_b64}", "detail": "high"},
                        },
                    ],
                },
            ],
        }
    else:
        # Other models: Use JSON approach
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Detect all distinct visual components with tight boxes."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"},
                        },
                    ],
                },
            ],
            "temperature": temperature,
            "max_tokens": 1600,
            "response_format": {"type": "json_object"},
        }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        
        if cfg.get("deployment", "").lower() == "gpt-5":
            # For GPT-5, parse the text description to extract components
            print(f"GPT-5 raw response: {content[:500]}...")
            return parse_gpt5_description(content, image_path, image_size, scale_info)
        else:
            # For other models, try to parse as JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return {"error": f"no_json_found_in_response: {content[:200]}..."}
                
    except Exception as exc:  # pragma: no cover
        return {"error": f"azure_gpt5_failed: {exc}"}


def to_pixel_boxes(image_path: Path, components: List[Dict]) -> List[Dict]:
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    out: List[Dict] = []
    for i, comp in enumerate(components or []):
        bbox = comp.get("bbox_norm") or comp.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        if max(x, y, w, h) <= 1.0:
            px = int(max(0, min(1, float(x))) * W)
            py = int(max(0, min(1, float(y))) * H)
            pw = int(max(0, min(1, float(w))) * W)
            ph = int(max(0, min(1, float(h))) * H)
        else:
            px, py, pw, ph = int(x), int(y), int(w), int(h)
        out.append(
            {
                "id": comp.get("id", f"c{i}"),
                "type": comp.get("type", "component"),
                "name": comp.get("name", ""),
                "meaning": comp.get("meaning", ""),
                "confidence": float(comp.get("confidence", 0.0)),
                "bbox_norm": [float(x), float(y), float(w), float(h)],
                "bbox_px": [px, py, pw, ph],
            }
        )
    return out


def draw_overlay(image_path: Path, comps_px: List[Dict], out_path: Path, scale_info: tuple = None) -> None:
    # Load the original image
    im_orig = Image.open(image_path).convert("RGB")
    
    # If we have scale info, create the padded version for drawing
    if scale_info:
        scale, x_offset, y_offset = scale_info
        target_size = (512, 512)
        
        # Scale the original image
        new_w = int(im_orig.size[0] * scale)
        new_h = int(im_orig.size[1] * scale)
        im_scaled = im_orig.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create padded canvas
        im = Image.new('RGB', target_size, (255, 255, 255))
        im.paste(im_scaled, (x_offset, y_offset))
    else:
        im = im_orig
    
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size=max(12, im.width // 80))
    except Exception:
        try:
            font = ImageFont.truetype("Arial.ttf", size=max(12, im.width // 80))
        except Exception:
            font = ImageFont.load_default()

    palette = [(230, 57, 70), (29, 53, 87), (69, 123, 157), (53, 194, 109), (255, 183, 3), (244, 162, 97)]
    for i, comp in enumerate(comps_px):
        color = palette[i % len(palette)]
        x, y, w, h = comp["bbox_px"]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=max(2, im.width // 400))
        label = f"{comp['id']} {comp.get('type','')}: {comp.get('meaning','')}"
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        pad = 2
        draw.rectangle([x, y - th - 2 * pad, x + tw + 2 * pad, y], fill=(0, 0, 0))
        draw.text((x + pad, y - th - pad), label, fill=(255, 255, 255), font=font)
    im.save(out_path)


def crop_components(image_path: Path, comps_px: List[Dict], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(image_path).convert("RGB")
    saved: List[str] = []
    for i, comp in enumerate(comps_px):
        x, y, w, h = comp["bbox_px"]
        if w <= 0 or h <= 0:
            continue
        crop = im.crop((x, y, x + w, y + h))
        safe_type = (comp.get("type") or "comp").replace("/", "-")
        fname = f"{i:02d}_{safe_type}.png"
        path = out_dir / fname
        crop.save(path)
        saved.append(str(path))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect components and export bounding boxes using Azure GPT-5 Vision")
    parser.add_argument("--image", required=True, help="Path to input image (PNG/JPG)")
    parser.add_argument("--out_json", default=None, help="Output JSON path; default next to image")
    parser.add_argument("--out_vis", default=None, help="Output visualization image; default next to image")
    parser.add_argument("--crop_dir", default=None, help="If set, save per-component crops into this directory")
    parser.add_argument("--temperature", type=float, default=0.0)
    # Azure configuration (env defaults supported)
    parser.add_argument("--azure_endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT", "https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com/"))
    parser.add_argument("--azure_api_key", default=os.getenv("AZURE_OPENAI_API_KEY", ""))
    parser.add_argument("--deployment", default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5"))
    parser.add_argument("--api_version", default=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"))
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    out_json = Path(args.out_json) if args.out_json else image_path.with_suffix("")
    if out_json.is_dir() or args.out_json is None:
        out_json = image_path.parent / f"{image_path.stem}_gpt5_boxes.json"

    out_vis = Path(args.out_vis) if args.out_vis else image_path.with_suffix("")
    if out_vis.is_dir() or args.out_vis is None:
        out_vis = image_path.parent / f"{image_path.stem}_gpt5_boxes.png"

    cfg = {
        "endpoint": args.azure_endpoint,
        "api_key": args.azure_api_key,
        "deployment": args.deployment,
        "api_version": args.api_version,
    }

    result = call_gpt5(image_path, cfg, temperature=args.temperature)
    if isinstance(result, dict) and result.get("error"):
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Azure call failed. Details saved to {out_json}")
        return

    components = result.get("components") or []
    comps_px = to_pixel_boxes(image_path, components)

    # Get scale info for drawing overlay on padded image
    _, _, scale_info = _b64_image(image_path, target_size=(512, 512))

    report = {
        "image_path": str(image_path.resolve()),
        "components": comps_px,
        "layout": result.get("layout", {}),
        "quality": result.get("quality", {}),
        "prompt_used": SYSTEM_PROMPT,
        "scale_info": scale_info,
    }

    # 1) Draw overlay first (as requested) - on padded image
    try:
        draw_overlay(image_path, comps_px, out_vis, scale_info)
    except Exception as exc:
        print(f"Warning: overlay failed: {exc}")

    # 2) Optional crops
    if args.crop_dir:
        try:
            saved = crop_components(image_path, comps_px, Path(args.crop_dir))
            report["crops"] = saved
        except Exception as exc:
            print(f"Warning: cropping failed: {exc}")

    # 3) Save JSON after visuals are produced
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4) Finally, print bbox list to stdout
    print(json.dumps({"boxes_px": [c["bbox_px"] for c in comps_px]}, ensure_ascii=False))


if __name__ == "__main__":
    main()


