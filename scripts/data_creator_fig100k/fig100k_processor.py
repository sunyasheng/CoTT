#!/usr/bin/env python3
"""
Fig100k Dataset Processor - Convert fig100k dataset to output format similar to azure_parallel_processor.py

Features:
1. Read fig100k dataset JSON files
2. Use GPT-4o to analyze image content as thinking
3. Use caption as diagram description
4. Generate training_data and judge_data format JSON output

Data Structure:
- Input: fig100k JSON format, containing caption, context, image path, etc.
- Output: Similar to azure_parallel_processor.py format, containing diagram-short, diagram-long, thinking-short, thinking-long

Usage:
python fig100k_processor.py --input_json /path/to/paper2fig_train.json --output_dir /path/to/output --workers 4
"""

import os
import sys
import json
import argparse
import logging
import base64
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fig100k_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Fig100kProcessor:
    """Fig100k Dataset Processor"""
    
    def __init__(self, max_workers: int = 4, skip_existing: bool = True):
        self.max_workers = max_workers
        self.skip_existing = skip_existing
        
        # Azure OpenAI Configuration - Load from environment variables
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com/")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if not self.api_key:
            logger.error("❌ Azure OpenAI API key not found. Please set AZURE_OPENAI_API_KEY environment variable.")
            raise ValueError("Azure OpenAI API key is required")
        
        # Control API request interval
        self.last_api_call_time = 0
        self.api_call_interval = 1.0  # 1 second interval to avoid rate limit
    
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
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Image encoding failed {image_path}: {e}")
            return ""
    
    def make_api_request_with_retry(self, url: str, headers: dict, payload: dict, 
                                  max_retries: int = 3, delay: float = 2.0, timeout: int = 60) -> dict:
        """API request with retry mechanism"""
        for attempt in range(max_retries):
            try:
                # Add request interval to avoid rate limit
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call_time
                if time_since_last_call < self.api_call_interval:
                    sleep_time = self.api_call_interval - time_since_last_call
                    time.sleep(sleep_time)
                
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                self.last_api_call_time = time.time()
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    logger.warning(f"⚠️ 401 error (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= 1.5
                        continue
                elif response.status_code == 429:
                    logger.warning(f"⚠️ 429 Rate Limit (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        time.sleep(wait_time)
                        continue
                else:
                    response.raise_for_status()
                    
            except Exception as e:
                logger.error(f"❌ API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 1.5
                else:
                    raise e
        
        return {"error": f"API request failed after {max_retries} attempts"}
    
    def analyze_image_with_gpt4o(self, image_path: str, caption: str = "", context: str = "") -> Dict[str, str]:
        """Use GPT-4o to analyze image content"""
        
        if not os.path.exists(image_path):
            logger.error(f"❌ Image file does not exist: {image_path}")
            return {"thinking_short": "", "thinking_long": ""}
        
        # Encode image
        base64_image = self.encode_image(image_path)
        if not base64_image:
            return {"thinking_short": "", "thinking_long": ""}
        
        # Build request URL
        url = f"{self.endpoint}openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        # Build prompts
        prompt_short = f"Please briefly analyze the content of this image. Image caption: {caption}"
        prompt_long = f"""Please provide a detailed analysis of this image content, including:
1. Overall structure and layout of the image
2. Text, numbers, symbols and other elements in the image
3. Visual elements like colors, shapes, lines in the image
4. Possible meanings or concepts expressed by the image
5. Relationship between the image and its caption

Image caption: {caption}
Context information: {context}

Please provide a detailed analysis."""
        
        results = {"thinking_short": "", "thinking_long": ""}
        
        # Generate short thinking
        try:
            payload_short = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_short},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            data_short = self.make_api_request_with_retry(url, headers, payload_short, max_retries=2, delay=1.0, timeout=30)
            if "error" not in data_short:
                results["thinking_short"] = data_short.get("choices", [{}])[0].get("message", {}).get("content", "")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate short thinking: {e}")
        
        # Generate long thinking
        try:
            payload_long = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_long},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                "max_tokens": 1500,
                "temperature": 0.1
            }
            
            data_long = self.make_api_request_with_retry(url, headers, payload_long, max_retries=2, delay=1.0, timeout=30)
            if "error" not in data_long:
                results["thinking_long"] = data_long.get("choices", [{}])[0].get("message", {}).get("content", "")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate long thinking: {e}")
        
        return results
    
    def process_single_item(self, item: Dict[str, Any], item_index: int) -> Dict[str, Any]:
        """Process single fig100k data item"""
        start_time = time.time()
        
        result = {
            "index": item_index,
            "status": "processing",
            "start_time": datetime.now().isoformat(),
            "error": None,
            "training_data": None,
            "judge_data": None
        }
        
        try:
            # Extract basic information from fig100k format
            figure_id = item.get("figure_id", "")
            captions = item.get("captions", [])
            captions_norm = item.get("captions_norm", [])
            aspect = item.get("aspect", 1.0)
            
            # Use the first caption as the main caption, or normalized caption if no captions
            if captions and len(captions) > 0:
                caption = captions[0]
                context = "; ".join(captions[1:]) if len(captions) > 1 else ""
            elif captions_norm and len(captions_norm) > 0:
                caption = captions_norm[0]
                context = "; ".join(captions_norm[1:]) if len(captions_norm) > 1 else ""
            else:
                result["status"] = "failed"
                result["error"] = "Missing captions"
                return result
            
            # Find the actual image file
            image_path = self.find_image_file(figure_id)
            
            if not figure_id:
                result["status"] = "failed"
                result["error"] = "Missing figure_id"
                return result
            
            if not image_path:
                result["status"] = "failed"
                result["error"] = f"Image file not found for figure_id: {figure_id}"
                return result
            
            logger.info(f"🔄 Processing item {item_index}: {caption[:50]}...")
            
            # Use GPT-4o to analyze image
            thinking_results = self.analyze_image_with_gpt4o(image_path, caption, context)
            
            if not thinking_results["thinking_short"] and not thinking_results["thinking_long"]:
                result["status"] = "failed"
                result["error"] = "GPT-4o analysis failed"
                return result
            
            # Generate diagram description (based on caption)
            diagram_short = caption
            diagram_long = f"{caption}\n\nContext: {context}" if context else caption
            
                # Build training_data format (matching diagram_training_data.json structure)
                training_data = {
                    "data_quality": "valid",
                    "quality_issues": [],
                    "stage1_input": {
                        "context": context if context else caption,
                        "caption": caption
                    },
                    "stage2_input": {
                        "diagram_description_short": diagram_short,
                        "diagram_description_long": diagram_long
                    },
                    "stage2_output": {
                        "thinking_short": thinking_results["thinking_short"],
                        "thinking_long": thinking_results["thinking_long"],
                        "image_path": image_path
                    }
                }
            
                # Build judge_data format (matching diagram_training_data.json structure)
                judge_data = {
                    "data_quality": "valid",
                    "quality_issues": [],
                    "stage1_input": {
                        "context": context if context else caption,
                        "caption": caption
                    },
                    "stage2_input": {
                        "diagram_description_short": diagram_short,
                        "diagram_description_long": diagram_long
                    },
                    "stage2_output": {
                        "thinking_short": thinking_results["thinking_short"],
                        "thinking_long": thinking_results["thinking_long"],
                        "image_path": image_path
                    }
                }
            
            result["training_data"] = training_data
            result["judge_data"] = judge_data
            result["status"] = "completed"
            
            logger.info(f"✅ Completed processing item {item_index}")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"❌ Error processing item {item_index}: {e}")
        
        result["end_time"] = datetime.now().isoformat()
        result["processing_time"] = time.time() - start_time
        
        return result
    
    def load_fig100k_data(self, json_path: str) -> List[Dict[str, Any]]:
        """Load fig100k dataset"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✅ Successfully loaded fig100k data: {len(data)} items")
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to load fig100k data: {e}")
            return []
    
    def process_parallel(self, input_json: str, output_dir: str, max_items: Optional[int] = None) -> Dict[str, Any]:
        """Process fig100k dataset in parallel"""
        start_time = time.time()
        
        # Load data
        data = self.load_fig100k_data(input_json)
        if not data:
            return {"error": "Unable to load fig100k data"}
        
        # Limit processing quantity
        if max_items and max_items < len(data):
            data = data[:max_items]
            logger.info(f"📊 Limited processing quantity to: {max_items}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Starting parallel processing of {len(data)} items using {self.max_workers} threads")
        
        # Parallel processing
        results = []
        failed_items = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.process_single_item, item, i): i 
                for i, item in enumerate(data)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["status"] == "failed":
                        failed_items.append(result)
                        logger.error(f"❌ Item {index} processing failed: {result.get('error', 'Unknown error')}")
                    else:
                        logger.info(f"✅ Item {index} processing completed")
                        
                except Exception as e:
                    logger.error(f"❌ Exception occurred while processing item {index}: {e}")
                    failed_items.append({
                        "index": index,
                        "status": "failed",
                        "error": str(e)
                    })
        
        # Merge all results
        all_training_data = []
        all_judge_data = []
        
        for result in results:
            if result["status"] == "completed":
                all_training_data.append(result["training_data"])
                all_judge_data.append(result["judge_data"])
        
        # Save results
        self.save_results(all_training_data, all_judge_data, results, output_path)
        
        # Generate statistics
        total_time = time.time() - start_time
        successful_items = len([r for r in results if r["status"] == "completed"])
        statistics = {
            "total_items": len(data),
            "successful_items": successful_items,
            "failed_items": len(failed_items),
            "total_training_items": len(all_training_data),
            "total_judge_items": len(all_judge_data),
            "total_processing_time": total_time,
            "average_time_per_item": total_time / len(data) if data else 0
        }
        
        logger.info(f"🎉 Parallel processing completed!")
        logger.info(f"📊 Statistics: {statistics}")
        
        return {
            "statistics": statistics,
            "results": results,
            "failed_items": failed_items,
            "all_training_data": all_training_data,
            "all_judge_data": all_judge_data
        }
    
    def save_results(self, training_data: List[Dict], judge_data: List[Dict], 
                    results: List[Dict], output_dir: Path):
        """Save processing results"""
        # Save training data
        training_file = output_dir / "fig100k_training_data.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        # Save judge data
        judge_file = output_dir / "fig100k_judge_data.json"
        with open(judge_file, 'w', encoding='utf-8') as f:
            json.dump(judge_data, f, ensure_ascii=False, indent=2)
        
        # Save processing report
        report_file = output_dir / "fig100k_processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Fig100k processing results saved to: {output_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fig100k Dataset Processor")
    parser.add_argument("--input_json", "-i", required=True, help="Input fig100k JSON file path")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel worker threads")
    parser.add_argument("--max_items", "-m", type=int, help="Maximum number of items to process (for testing)")
    
    args = parser.parse_args()
    
    # Create processor
    processor = Fig100kProcessor(max_workers=args.workers)
    
    # Start processing
    result = processor.process_parallel(args.input_json, args.output_dir, max_items=args.max_items)
    
    if "error" in result:
        logger.error(f"❌ Processing failed: {result['error']}")
        sys.exit(1)
    else:
        logger.info("🎉 All processing completed!")


if __name__ == "__main__":
    main()


# python fig100k_processor.py --input_json /blob/yasheng/paper2fig_train.json --output_dir ./paper2fig_train_output --workers 4 --max_items 10