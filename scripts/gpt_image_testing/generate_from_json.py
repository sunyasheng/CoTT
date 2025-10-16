#!/usr/bin/env python3
"""
从JSON文件读取thinking_short并使用Azure OpenAI生成对应的图片
"""

import requests
import json
import base64
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from pathlib import Path

# 加载 .env 文件中的环境变量
load_dotenv()


class AzureOpenAIImageGenerator:
    def __init__(self, endpoint: str, api_key: str):
        """
        初始化Azure OpenAI图片生成器
        
        Args:
            endpoint: Azure OpenAI API端点
            api_key: API密钥
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'api-key': api_key
        }
    
    def generate_image(
        self, 
        prompt: str, 
        size: str = "1024x1024",
        quality: str = "medium",
        n: int = 1
    ) -> Dict[str, Any]:
        """
        生成图片
        
        Args:
            prompt: 图片描述提示词
            size: 图片尺寸 ("1024x1024", "1536x1024", "1024x1536")
            quality: 图片质量 ("low", "medium", "high", "auto")
            n: 生成图片数量 (1-10)
        
        Returns:
            包含生成结果的字典
        """
        url = f"{self.endpoint}/openai/deployments/gpt-image-1/images/generations?api-version=2025-04-01-preview"
        
        payload = {
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": n
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code != 200:
                print(f"错误响应: {response.text}")
            
            response.raise_for_status()
            
            result = response.json()
            return {
                "success": True,
                "data": result,
                "error": None
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "data": None,
                "error": f"JSON解析错误: {str(e)}"
            }
    
    def save_image(
        self, 
        image_data: Dict[str, Any], 
        output_dir: str = "generated_images",
        filename_prefix: str = "generated_image"
    ) -> Optional[str]:
        """
        保存生成的图片到本地
        
        Args:
            image_data: 图片数据字典
            output_dir: 输出目录
            filename_prefix: 文件名前缀
        
        Returns:
            保存的文件路径，如果失败返回None
        """
        if not image_data.get("success") or not image_data.get("data"):
            print("没有有效的图片数据可保存")
            return None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            if "data" in image_data["data"] and len(image_data["data"]["data"]) > 0:
                img_data = image_data["data"]["data"][0]
                
                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename_prefix}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                
                if "url" in img_data:
                    # 从URL下载图片
                    img_response = requests.get(img_data["url"])
                    img_response.raise_for_status()
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)
                elif "b64_json" in img_data:
                    # 从base64数据保存图片
                    image_bytes = base64.b64decode(img_data["b64_json"])
                    with open(filepath, 'wb') as f:
                        f.write(image_bytes)
                else:
                    print(f"不支持的图片格式: {list(img_data.keys())}")
                    return None
                
                print(f"图片已保存到: {filepath}")
                return filepath
            else:
                print("无法获取图片数据")
                return None
            
        except Exception as e:
            print(f"保存图片时出错: {str(e)}")
            return None


def load_training_data(json_path: str) -> List[Dict[str, Any]]:
    """
    加载训练数据JSON文件
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        训练数据列表
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"读取JSON文件失败: {str(e)}")
        return []


def generate_images_from_json(
    json_path: str,
    output_dir: str = "generated_diagrams",
    size: str = "1024x1024",
    quality: str = "medium"
):
    """
    从JSON文件读取thinking_short并生成对应的图片
    
    Args:
        json_path: JSON文件路径
        output_dir: 输出目录
        size: 图片尺寸
        quality: 图片质量
    """
    # 从环境变量读取Azure OpenAI配置
    ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    
    if not ENDPOINT or not API_KEY:
        raise ValueError(
            "请在 .env 文件中设置 AZURE_OPENAI_ENDPOINT 和 AZURE_OPENAI_API_KEY\n"
            "你可以复制 .env.example 为 .env 并填入你的配置"
        )
    
    # 创建图片生成器实例
    generator = AzureOpenAIImageGenerator(ENDPOINT, API_KEY)
    
    # 加载训练数据
    print(f"正在加载: {json_path}")
    training_data = load_training_data(json_path)
    
    if not training_data:
        print("没有找到训练数据")
        return
    
    print(f"找到 {len(training_data)} 条数据\n")
    
    # 为每条数据生成图片
    results = []
    for idx, item in enumerate(training_data, 1):
        print(f"\n{'='*60}")
        print(f"处理第 {idx}/{len(training_data)} 条数据")
        print(f"{'='*60}")
        
        # 获取thinking_short
        thinking_short = item.get("stage2_output", {}).get("thinking_short", "")
        
        if not thinking_short:
            print(f"⚠️  第 {idx} 条数据没有 thinking_short，跳过")
            results.append({
                "index": idx,
                "success": False,
                "error": "没有thinking_short"
            })
            continue
        
        # 显示提示词（截断显示）
        display_prompt = thinking_short[:200] + "..." if len(thinking_short) > 200 else thinking_short
        print(f"\n提示词: {display_prompt}\n")
        
        # 生成图片
        print("正在生成图片...")
        result = generator.generate_image(
            prompt=thinking_short,
            size=size,
            quality=quality
        )
        
        if result["success"]:
            print("✓ 图片生成成功!")
            
            # 保存图片
            # 使用原始JSON文件名和索引作为文件名前缀
            json_basename = Path(json_path).stem
            filename_prefix = f"{json_basename}_item{idx}"
            
            saved_path = generator.save_image(
                result,
                output_dir=output_dir,
                filename_prefix=filename_prefix
            )
            
            if saved_path:
                results.append({
                    "index": idx,
                    "success": True,
                    "image_path": saved_path,
                    "prompt": thinking_short
                })
            else:
                results.append({
                    "index": idx,
                    "success": False,
                    "error": "保存失败"
                })
        else:
            print(f"✗ 图片生成失败: {result['error']}")
            results.append({
                "index": idx,
                "success": False,
                "error": result['error']
            })
    
    # 输出总结
    print(f"\n\n{'='*60}")
    print("生成完成！")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r.get("success"))
    print(f"成功: {success_count}/{len(results)}")
    print(f"失败: {len(results) - success_count}/{len(results)}")
    
    # 保存结果摘要
    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果摘要已保存到: {summary_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="从JSON文件的thinking_short生成图片"
    )
    parser.add_argument(
        "json_path",
        help="训练数据JSON文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        default="generated_diagrams",
        help="输出目录 (默认: generated_diagrams)"
    )
    parser.add_argument(
        "-s", "--size",
        default="1024x1024",
        choices=["1024x1024", "1536x1024", "1024x1536"],
        help="图片尺寸 (默认: 1024x1024)"
    )
    parser.add_argument(
        "-q", "--quality",
        default="medium",
        choices=["low", "medium", "high", "auto"],
        help="图片质量 (默认: medium)"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.json_path):
        print(f"错误: 文件不存在: {args.json_path}")
        return 1
    
    # 生成图片
    generate_images_from_json(
        json_path=args.json_path,
        output_dir=args.output,
        size=args.size,
        quality=args.quality
    )
    
    return 0


if __name__ == "__main__":
    exit(main())

