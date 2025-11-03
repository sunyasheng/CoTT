#!/usr/bin/env python3
"""
Diagram Parse Graph Creator
根据论文 "A Diagram Is Worth A Dozen Images" (arXiv:1603.07396) 的 Diagram Parse Graph (DPG) 格式
来标注图表图片

Usage:
    python diagram_parse_graph_creator.py --image path/to/image.png [--output output.json]
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not found. Please install it with: pip install openai")
    sys.exit(1)

# Load environment variables
load_dotenv()

# DPG System Prompt based on the paper
DPG_SYSTEM_PROMPT = """You are an expert at analyzing diagrams and extracting Diagram Parse Graphs (DPG) according to the paper "A Diagram Is Worth A Dozen Images" (arXiv:1603.07396).

A Diagram Parse Graph models diagrams using four types of constituents:
1. **Blobs (Illustrations)**: Visual illustrations, shapes, diagrams, charts, or any non-text visual elements
2. **Text Boxes**: Textual elements in the diagram
3. **Arrows**: Arrows or lines connecting elements
4. **Arrow Heads**: The arrowhead part of arrows

And ten types of relationships (R1-R10):
- **R1 (Intra-Object Label)**: A text box naming the entire object
- **R2 (Intra-Object Region Label)**: A text box referring to a region within an object
- **R3 (Intra-Object Linkage)**: A text box referring to a region within an object via an arrow
- **R4 (Inter-Object Linkage)**: Two objects related to one another via an arrow
- **R5 (Arrow Head Assignment)**: An arrow head associated to an arrow tail
- **R6 (Arrow Descriptor)**: A text box describing a process that an arrow refers to
- **R7 (Image Title)**: The title of the entire image
- **R8 (Image Section Title)**: Text box that serves as a title for a section of the image
- **R9 (Image Caption)**: A text box that adds information about the entire image, but does not serve as the image title
- **R10 (Image Misc)**: Decorative elements in the diagram

Your task is to analyze the diagram and extract all constituents and their relationships in DPG format."""

DPG_USER_PROMPT = """Analyze this diagram and extract a complete Diagram Parse Graph (DPG) according to the paper specifications.

**STEP 1: Understand the Diagram's Main Meaning and Storyline**

First, carefully analyze the diagram to understand:
1. **What is the main purpose or message of this diagram?** (What concept, process, or system is it illustrating?)
2. **What is the overall storyline or flow?** (How does information, data, or concepts flow through the diagram?)
3. **What are the key components and their roles?** (What are the main building blocks and what do they do?)
4. **What is the input and output?** (What goes in and what comes out?)

This understanding will help you identify which elements are most important for expressing the diagram's meaning.

**STEP 2: Identify Constituents (Prioritize Elements that Express the Main Meaning)**

Based on your understanding from Step 1, identify the elements that are most important for expressing the diagram's meaning. If there are many similar elements, focus on the most important ones:

1. **Blobs (Illustrations)**: Visual elements like boxes, circles, charts, diagrams, images, shapes, etc.
   - Prioritize: Main components, key visual elements that represent important concepts
   - If there are many similar elements, focus on the most important ones
   - Provide clear descriptions that identify what each blob represents

2. **Text Boxes**: Text elements that are important for understanding the diagram
   - Prioritize: Titles, section headers, component names, key labels, and text that describes the main process
   - Include: Text on or near arrows that describes processes, transformations, or relationships
   - You can omit: Index labels (like subscripts, superscripts, or numbered indices) and detailed data entries if there are many similar ones

3. **Arrows**: Arrows and lines that show important connections and data flow
   - Prioritize: Arrows that represent the main data flow, key connections between major components
   - If there are many arrows pointing to the same target, you can represent them as a single conceptual arrow
   - Include: Bidirectional arrows should be represented appropriately

4. **Arrow Heads**: Arrowhead components for the arrows you identified

For each constituent, assign a unique ID and describe:
- For Blobs: type, position, visual properties, and a clear description of what the blob represents
- For Text Boxes: the exact text content, position, font style if notable
- For Arrows: source and target, direction, type (straight, curved, etc.)
- For Arrow Heads: which arrow they belong to

**STEP 3: Identify Relationships (Focus on Meaningful Connections)**

Based on your understanding of the diagram's meaning, identify relationships that are important for expressing the diagram's structure and flow:

- **R1**: Text boxes that name entire objects (e.g., component names, module names, or object labels)
- **R2**: Text boxes that label regions within objects (e.g., labels inside boxes, axis labels) - only if important for understanding
- **R3**: Text boxes that label regions via arrows (e.g., labels on arrows that refer to specific regions) - only if important
- **R4**: Arrows connecting objects (e.g., data flow arrows) - focus on the main flow
- **R5**: Arrow heads assigned to arrows
- **R6**: Text boxes describing what arrows represent (e.g., process names, transformation labels, or relationship descriptions)
- **R7**: Main title of the entire diagram
- **R8**: Section titles - all important section titles that organize the diagram into logical parts
- **R9**: Caption text that describes the diagram
- **R10**: Decorative elements (borders, frames, etc.) - only if they are significant

Return your analysis in the following JSON format:

{
  "constituents": {
    "blobs": [
      {
        "id": "blob_1",
        "type": "rectangle/circle/ellipse/chart/diagram/image/shape",
        "description": "description of the blob",
        "position": {"x": 0, "y": 0, "width": 100, "height": 50},
        "visual_properties": {"color": "blue", "style": "solid/dashed"}
      }
    ],
    "text_boxes": [
      {
        "id": "text_1",
        "content": "exact text content",
        "position": {"x": 0, "y": 0},
        "font_style": "bold/normal/italic",
        "font_size": "large/medium/small"
      }
    ],
    "arrows": [
      {
        "id": "arrow_1",
        "source": "blob_1 or text_1",
        "target": "blob_2 or text_2",
        "direction": "left-to-right/right-to-left/top-to-bottom/bottom-to-top/diagonal",
        "type": "straight/curved/dashed",
        "style": "solid/dashed/dotted"
      }
    ],
    "arrow_heads": [
      {
        "id": "arrowhead_1",
        "arrow_id": "arrow_1",
        "position": {"x": 0, "y": 0},
        "direction": "right/left/up/down"
      }
    ]
  },
  "relationships": {
    "R1_intra_object_label": [
      {
        "text_box_id": "text_1",
        "object_id": "blob_1",
        "description": "text box names the entire object"
      }
    ],
    "R2_intra_object_region_label": [
      {
        "text_box_id": "text_2",
        "object_id": "blob_1",
        "region": "region description",
        "description": "text box labels a region within the object"
      }
    ],
    "R3_intra_object_linkage": [
      {
        "text_box_id": "text_3",
        "object_id": "blob_1",
        "arrow_id": "arrow_1",
        "region": "region description",
        "description": "text box labels a region via arrow"
      }
    ],
    "R4_inter_object_linkage": [
      {
        "arrow_id": "arrow_1",
        "source_object_id": "blob_1",
        "target_object_id": "blob_2",
        "description": "arrow connects two objects"
      }
    ],
    "R5_arrow_head_assignment": [
      {
        "arrow_id": "arrow_1",
        "arrowhead_id": "arrowhead_1",
        "description": "arrow head assigned to arrow"
      }
    ],
    "R6_arrow_descriptor": [
      {
        "text_box_id": "text_4",
        "arrow_id": "arrow_1",
        "description": "text box describes what the arrow represents"
      }
    ],
    "R7_image_title": [
      {
        "text_box_id": "text_5",
        "description": "title of the entire image"
      }
    ],
    "R8_image_section_title": [
      {
        "text_box_id": "text_6",
        "section": "section name",
        "description": "title for a section of the image"
      }
    ],
    "R9_image_caption": [
      {
        "text_box_id": "text_7",
        "description": "caption adding information about the entire image"
      }
    ],
    "R10_image_misc": [
      {
        "blob_id": "blob_10",
        "description": "decorative element"
      }
    ]
  }
}

**Important Guidelines:**
- Focus on elements that are essential for understanding the diagram's main meaning and storyline
- If there are many similar elements (like multiple index labels or similar data entries), prioritize the most important ones or represent them conceptually
- Ensure that the identified elements can tell the complete story of what the diagram represents
- Use approximate positions if exact coordinates are not available
- The goal is to capture the diagram's semantic structure, not every single visual detail"""


class DiagramParseGraphCreator:
    """创建 Diagram Parse Graph 的工具类"""
    
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, api_type: str = "openai"):
        """初始化
        
        Args:
            api_key: API key (OpenAI or Azure OpenAI)
            api_base: API base URL (for Azure OpenAI)
            api_type: "openai" or "azure"
        """
        self.api_type = api_type
        
        if api_type == "azure":
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            self.api_base = api_base or os.getenv("AZURE_OPENAI_ENDPOINT", "")
            self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
            self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            
            if not self.api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not found. Please set it in environment or pass as argument.")
            if not self.api_base:
                raise ValueError("AZURE_OPENAI_ENDPOINT not found. Please set it in environment or pass as argument.")
            
            # Azure OpenAI uses different configuration
            # base_url format: https://{endpoint}/openai/deployments/{deployment}
            endpoint_url = self.api_base.rstrip('/')
            if not endpoint_url.endswith('/openai'):
                endpoint_url = f"{endpoint_url}/openai"
            
            # The base_url should point to the deployment
            deployment_base_url = f"{endpoint_url}/deployments/{self.deployment}"
            
            # For Azure OpenAI with OpenAI SDK, we need to add default_query for api-version
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.api_base.rstrip('/')
            )
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found. Please set it in environment or pass as argument.")
            self.client = OpenAI(api_key=self.api_key)
    
    def encode_image(self, image_path: str) -> str:
        """将图片编码为base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to encode image: {e}")
    
    def analyze_diagram(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """分析图表并生成 Diagram Parse Graph"""
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"📸 Reading image: {image_path}")
        base64_image = self.encode_image(image_path)
        
        print("🤖 Analyzing diagram with GPT-4o...")
        
        try:
            if self.api_type == "azure":
                # Azure OpenAI uses deployment name as model
                response = self.client.chat.completions.create(
                    model=self.deployment,  # Azure uses deployment name
                    messages=[
                        {
                            "role": "system",
                            "content": DPG_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": DPG_USER_PROMPT
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.1
                )
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": DPG_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": DPG_USER_PROMPT
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.1
                )
            
            content = response.choices[0].message.content
            print("✅ Received response from GPT-4o")
            
            # Try to parse JSON from response
            try:
                # First try direct JSON parsing
                dpg_result = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    dpg_result = json.loads(json_match.group(1))
                else:
                    # Try to find JSON object in the text
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        dpg_result = json.loads(json_match.group(0))
                    else:
                        raise ValueError("Could not extract JSON from response")
            
            # Add metadata
            result = {
                "image_path": str(image_path),
                "model": "gpt-4o",
                "dpg": dpg_result
            }
            
            # Save to file if output path specified
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"💾 Saved DPG to: {output_path}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error analyzing diagram: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Create Diagram Parse Graph (DPG) from a diagram image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diagram_parse_graph_creator.py --image diagram.png
  python diagram_parse_graph_creator.py --image diagram.png --output dpg_result.json
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to the diagram image file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (default: <image_name>_dpg.json)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (default: from OPENAI_API_KEY or AZURE_OPENAI_API_KEY env var)'
    )
    
    parser.add_argument(
        '--api-base',
        type=str,
        default=None,
        help='API base URL for Azure OpenAI (default: from AZURE_OPENAI_ENDPOINT env var)'
    )
    
    parser.add_argument(
        '--api-type',
        type=str,
        choices=['openai', 'azure'],
        default='openai',
        help='API type: openai or azure (default: openai)'
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        image_path = Path(args.image)
        args.output = str(image_path.parent / f"{image_path.stem}_dpg.json")
    
    try:
        creator = DiagramParseGraphCreator(
            api_key=args.api_key,
            api_base=args.api_base,
            api_type=args.api_type
        )
        result = creator.analyze_diagram(args.image, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Diagram Parse Graph Summary")
        print("="*60)
        
        dpg = result.get("dpg", {})
        constituents = dpg.get("constituents", {})
        relationships = dpg.get("relationships", {})
        
        print(f"\nConstituents:")
        print(f"  - Blobs: {len(constituents.get('blobs', []))}")
        print(f"  - Text Boxes: {len(constituents.get('text_boxes', []))}")
        print(f"  - Arrows: {len(constituents.get('arrows', []))}")
        print(f"  - Arrow Heads: {len(constituents.get('arrow_heads', []))}")
        
        print(f"\nRelationships:")
        for rel_type, rels in relationships.items():
            if rels:
                print(f"  - {rel_type}: {len(rels)}")
        
        print(f"\n✅ Complete DPG saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

