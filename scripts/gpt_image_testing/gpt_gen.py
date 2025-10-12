#!/usr/bin/env python3
"""
Azure OpenAI Image Generation API 示例
使用DALL-E模型生成图片
"""

import requests
import json
import base64
import os
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

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
            size: 图片尺寸 ("1024x1024", "1792x1024", "1024x1792")
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
    
    def save_image(self, image_data: Dict[str, Any], output_dir: str = "generated_images") -> Optional[str]:
        """
        保存生成的图片到本地
        
        Args:
            image_data: 图片数据字典
            output_dir: 输出目录
        
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
                filename = f"generated_image_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                
                if "url" in img_data:
                    # 从URL下载图片
                    img_response = requests.get(img_data["url"])
                    img_response.raise_for_status()
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)
                elif "b64_json" in img_data:
                    # 从base64数据保存图片
                    import base64
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


def main():
    """主函数 - 示例用法"""
    
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
    
    # 示例1: 生成一张简单的图片
    print("=== 示例1: 生成简单图片 ===")
    prompt1 = "A beautiful sunset over a mountain landscape with a lake in the foreground"
    result1 = generator.generate_image(prompt1)
    
    if result1["success"]:
        print("图片生成成功!")
        # 根据实际响应格式获取图片数据
        if "data" in result1["data"] and len(result1["data"]["data"]) > 0:
            img_data = result1["data"]["data"][0]
            if "url" in img_data:
                print(f"图片URL: {img_data['url']}")
            elif "b64_json" in img_data:
                print("图片以base64格式返回")
            else:
                print(f"图片数据格式: {list(img_data.keys())}")
        else:
            print("无法获取图片数据")
        
        # 保存图片到本地
        saved_path = generator.save_image(result1)
        if saved_path:
            print(f"图片已保存到: {saved_path}")
    else:
        print(f"图片生成失败: {result1['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # 示例2: 生成高质量图片
    print("=== 示例2: 生成高质量图片 ===")
    prompt2 = "A futuristic city skyline at night with neon lights and flying cars"
    result2 = generator.generate_image(
        prompt=prompt2,
        size="1536x1024",
        quality="high"
    )
    
    if result2["success"]:
        print("高质量图片生成成功!")
        # 根据实际响应格式获取图片数据
        if "data" in result2["data"] and len(result2["data"]["data"]) > 0:
            img_data = result2["data"]["data"][0]
            if "url" in img_data:
                print(f"图片URL: {img_data['url']}")
            elif "b64_json" in img_data:
                print("图片以base64格式返回")
            else:
                print(f"图片数据格式: {list(img_data.keys())}")
        else:
            print("无法获取图片数据")
        
        # 保存图片到本地
        saved_path = generator.save_image(result2)
        if saved_path:
            print(f"图片已保存到: {saved_path}")
    else:
        print(f"图片生成失败: {result2['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # 示例3: 生成多张图片
    print("=== 示例3: 生成多张图片 ===")
    prompt3 = "A cute cartoon cat sitting on a rainbow"
    result3 = generator.generate_image(
        prompt=prompt3,
        n=3  # 生成3张图片
    )
    
    if result3["success"]:
        # 根据实际响应格式处理多张图片
        if "data" in result3["data"] and len(result3["data"]["data"]) > 0:
            print(f"成功生成 {len(result3['data']['data'])} 张图片!")
            for i, img_data in enumerate(result3['data']['data']):
                if "url" in img_data:
                    print(f"图片 {i+1} URL: {img_data['url']}")
                elif "b64_json" in img_data:
                    print(f"图片 {i+1} 以base64格式返回")
                else:
                    print(f"图片 {i+1} 数据格式: {list(img_data.keys())}")
        else:
            print("无法获取图片信息")
    else:
        print(f"图片生成失败: {result3['error']}")


if __name__ == "__main__":
    main()
