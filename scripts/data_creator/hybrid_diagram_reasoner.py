#!/usr/bin/env python3
"""
Hybrid Diagram Figure Reasoner - 支持多API源的智能图表分析器

功能：
1. 支持切换不同的GPT-4o接口源（Papyrus 和 Azure OpenAI）
2. 提取所有图片的caption
3. 使用GPT判断哪些是diagram
4. 只分析被GPT识别为diagram的图片
5. 生成绘图指令

支持的API源：
- Papyrus (Microsoft内部API，默认)
- Azure OpenAI (备选)
"""

import os
import re
import json
import requests
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from prompts_template.prompt_manager import prompt_manager
from langchain.schema import Document
HAS_LANGCHAIN = True

# SentenceTransformer imports for local embeddings (强制使用，避免Azure API限制)
try:
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_SENTENCE_TRANSFORMERS = True
    print("✅ SentenceTransformer 已就绪 - 将强制使用本地模型避免API限制")
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("❌ SentenceTransformer 未安装！")
    print("🚫 我们强制使用本地模型以避免Azure API rate限制")
    print("📦 请安装: pip install sentence-transformers torch")

from dotenv import load_dotenv
HAS_DOTENV = True

# Azure Identity imports for Papyrus
try:
    from azure.identity import DefaultAzureCredential, AzureCliCredential, ManagedIdentityCredential
    HAS_AZURE_IDENTITY = True
except ImportError:
    HAS_AZURE_IDENTITY = False
    print("⚠️ Azure Identity not available. Papyrus API will not work.")


class APISource(Enum):
    """API源枚举"""
    AZURE_OPENAI = "azure_openai"
    PAPYRUS = "papyrus"


class HybridDiagramReasoner:
    """混合图表分析器，支持多API源"""
    
    def __init__(self, api_source: APISource = APISource.PAPYRUS):
        self.api_source = api_source
        self.last_api_call_time = 0
        self.api_call_interval = 1.0  # 1秒间隔，避免rate limit
        self.load_env_vars()
        self.setup_api_config()
    
    def load_env_vars(self):
        """加载环境变量"""
        # 尝试从多个位置加载 .env 文件
        env_paths = [
            Path(__file__).parent / ".env",
            Path(__file__).parent.parent / ".env",
            Path(__file__).parent.parent.parent / "CoTT" / ".env",
            Path(__file__).parent.parent.parent / "CoTT" / ".env_old",
            Path("/Users/suny0a/Proj/MM-Reasoning/CoTT/.env"),  # 绝对路径
            Path("/home/t2vg-a100-G2-0/yasheng/CoTT/.env"),  # 服务器路径
        ]
        
        if HAS_DOTENV:
            for env_path in env_paths:
                if env_path.exists():
                    print(f"📄 加载环境变量: {env_path}")
                    load_dotenv(env_path)
                    # 验证关键环境变量是否加载成功
                    api_key = os.getenv("AZURE_OPENAI_API_KEY")
                    if api_key:
                        print(f"✅ Azure OpenAI API Key 已加载: {api_key[:20]}...")
                    else:
                        print("❌ Azure OpenAI API Key 未找到")
                    return True
        else:
            # 简单的环境变量加载
            for env_path in env_paths:
                if env_path.exists():
                    print(f"📄 从文件加载环境变量: {env_path}")
                    with open(env_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                os.environ[key] = value
                    
                    # 验证关键环境变量是否加载成功
                    api_key = os.getenv("AZURE_OPENAI_API_KEY")
                    if api_key:
                        print(f"✅ Azure OpenAI API Key 已加载: {api_key[:20]}...")
                    else:
                        print("❌ Azure OpenAI API Key 未找到")
                    return True
        
        print("⚠️ 未找到环境变量文件")
        return False
    
    def setup_api_config(self):
        """设置API配置"""
        if self.api_source == APISource.AZURE_OPENAI:
            self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com/")
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
            self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
            self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            
            if not self.api_key:
                print("❌ Azure OpenAI API key not found")
                return False
            
            print(f"🔗 使用 Azure OpenAI API")
            print(f"   Endpoint: {self.endpoint}")
            print(f"   Deployment: {self.deployment}")
            
        elif self.api_source == APISource.PAPYRUS:
            if not HAS_AZURE_IDENTITY:
                print("❌ Azure Identity not available for Papyrus API")
                return False
            
            # 从环境变量或默认值获取Papyrus配置
            self.papyrus_endpoint = os.getenv("PAPYRUS_ENDPOINT", "https://WestUS2Large.papyrus.binginternal.com/chat/completions")
            self.verify_scope = os.getenv("PAPYRUS_VERIFY_SCOPE", "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default")
            self.client_id = os.getenv("PAPYRUS_CLIENT_ID", "d5702df1-96d9-4195-83a3-e44d8b0a0601")
            
            # 尝试不同的认证方式
            self.access_token = None
            self.setup_papyrus_auth()
            
            if not self.access_token:
                print("❌ Failed to get Papyrus access token")
                return False
            
            print(f"🔗 使用 Papyrus API")
            print(f"   Endpoint: {self.papyrus_endpoint}")
            print(f"   Access token: {self.access_token[:20]}...")
        
        return True
    
    def setup_papyrus_auth(self):
        """设置Papyrus认证"""
        # 优先尝试ManagedIdentityCredential（与papyrus_on_vm_2.py保持一致）
        try:
            print("🔐 尝试使用 Managed Identity 认证...")
            cred = ManagedIdentityCredential(client_id=self.client_id)
            self.access_token = cred.get_token(self.verify_scope).token
            print("✅ Managed Identity 认证成功")
            return True
        except Exception as e:
            print(f"❌ Managed Identity 认证失败: {e}")
        
        try:
            # 尝试使用DefaultAzureCredential
            print("🔐 尝试使用 Default Azure 认证...")
            cred = DefaultAzureCredential()
            self.access_token = cred.get_token(self.verify_scope).token
            print("✅ Default Azure 认证成功")
            return True
        except Exception as e:
            print(f"❌ Default Azure 认证失败: {e}")
        
        try:
            # 最后尝试使用AzureCliCredential
            print("🔐 尝试使用 Azure CLI 认证...")
            cred = AzureCliCredential()
            self.access_token = cred.get_token(self.verify_scope).token
            print("✅ Azure CLI 认证成功")
            return True
        except Exception as e:
            print(f"❌ Azure CLI 认证失败: {e}")
        
        return False
    
    def get_api_headers(self) -> Dict[str, str]:
        """获取API请求头"""
        if self.api_source == APISource.AZURE_OPENAI:
            return {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
        elif self.api_source == APISource.PAPYRUS:
            return {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "papyrus-model-name": os.getenv("PAPYRUS_MODEL_NAME", "gpt4ovision-batch"),
                "papyrus-timeout-ms": os.getenv("PAPYRUS_TIMEOUT_MS", "30000"),
                "papyrus-quota-id": os.getenv("PAPYRUS_QUOTA_ID", "msftaicopilot/windowsdata"),
            }
        return {}
    
    def get_api_url(self) -> str:
        """获取API请求URL"""
        if self.api_source == APISource.AZURE_OPENAI:
            return f"{self.endpoint}openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        elif self.api_source == APISource.PAPYRUS:
            return self.papyrus_endpoint
        return ""
    
    def make_api_request_with_retry(self, payload: Dict, max_retries: int = 3, delay: float = 2.0) -> Dict:
        """带重试机制的API请求"""
        for attempt in range(max_retries):
            try:
                # 添加请求间隔，避免rate limit
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call_time
                if time_since_last_call < self.api_call_interval:
                    sleep_time = self.api_call_interval - time_since_last_call
                    print(f"   ⏳ 等待 {sleep_time:.1f} 秒避免rate limit...")
                    time.sleep(sleep_time)
                
                url = self.get_api_url()
                headers = self.get_api_headers()
                
                # 如果是Papyrus API且token可能过期，重新获取token
                if self.api_source == APISource.PAPYRUS and attempt > 0:
                    print(f"   🔄 重试 {attempt + 1}/{max_retries}，重新获取token...")
                    self.setup_papyrus_auth()
                    headers = self.get_api_headers()
                
                response = requests.post(url, headers=headers, json=payload, timeout=180)
                self.last_api_call_time = time.time()  # 更新最后调用时间
                
                if response.status_code == 401:
                    print(f"   ⚠️ 401错误 (尝试 {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        print(f"   ⏳ 等待 {delay} 秒后重试...")
                        time.sleep(delay)
                        delay *= 1.5  # 指数退避
                        continue
                    else:
                        response.raise_for_status()
                elif response.status_code == 429:
                    print(f"   ⚠️ 429 Rate Limit (尝试 {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # 指数退避
                        print(f"   ⏳ 等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                else:
                    response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"   ❌ API请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"   ⏳ 等待 {delay} 秒后重试...")
                    time.sleep(delay)
                    delay *= 1.5
                else:
                    raise e
        
        return {"error": f"API request failed after {max_retries} attempts"}
    
    def extract_all_figures_from_markdown(self, markdown_content: str) -> List[Dict]:
        """提取所有图片信息，不做任何过滤"""
        figures = []
        seen_images = set()  # 避免重复提取
        
        # 找到附录开始的位置（排除附录内容）
        lines = markdown_content.split('\n')
        appendix_start_idx = len(lines)  # 默认没有附录
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            # 检查是否是附录标题
            if (line_lower.startswith('# appendix') or 
                line_lower.startswith('## appendix') or
                line_lower.startswith('### appendix') or
                line_lower.startswith('# supplementary') or
                line_lower.startswith('## supplementary') or
                line_lower.startswith('### supplementary') or
                'appendix' in line_lower and line_lower.startswith('#')):
                appendix_start_idx = i
                break
        
        # 只处理正文部分（排除附录）
        main_content = '\n'.join(lines[:appendix_start_idx])
        
        # 查找markdown格式的图片引用: ![](path) 或 ![alt](path)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+\.(?:jpg|jpeg|png|gif|bmp|svg))\)'
        image_matches = re.findall(image_pattern, main_content, re.IGNORECASE)
        
        # 处理每个图片引用
        for alt_text, image_path in image_matches:
            # 避免重复处理
            if image_path in seen_images:
                continue
            seen_images.add(image_path)
            
            # 查找这个图片后面的caption
            caption = ""
            image_pos = main_content.find(f"![{alt_text}]({image_path})")
            if image_pos != -1:
                # 查找图片后的文本作为caption
                after_image = main_content[image_pos + len(f"![{alt_text}]({image_path})"):]
                # 查找下一行或段落作为caption
                lines_after = after_image.split('\n')
                for line in lines_after:
                    line = line.strip()
                    if line and not line.startswith('![') and not line.startswith('#'):
                        # 检查是否是Figure开头的caption
                        if line.lower().startswith('figure'):
                            caption = line
                            break
                        # 或者取第一行非空文本作为caption
                        elif not caption:
                            caption = line
                        # 如果遇到下一个图片或标题，停止
                        if line.startswith('![') or line.startswith('#'):
                            break
            
            figures.append({
                'id': f"figure_{len(figures) + 1}",
                'src': image_path,
                'caption': caption or alt_text or f"Figure {len(figures) + 1}",
                'alt_text': alt_text
            })
        
        return figures
    
    def classify_figures_with_gpt(self, figures: List[Dict], paper_title: str = "") -> List[Dict]:
        """使用GPT判断哪些图片是diagram"""
        
        if not self.setup_api_config():
            print("❌ API配置失败")
            return []
        
        # 构建所有图片的caption信息
        figure_info = []
        for i, fig in enumerate(figures, 1):
            figure_info.append(f"Figure {i}: {fig['caption']}")
        
        figures_text = "\n".join(figure_info)
        
        # 使用prompt模板
        prompt = prompt_manager.get_figure_classification_prompt(paper_title, figures_text)

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        try:
            data = self.make_api_request_with_retry(payload, max_retries=3, delay=2.0)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # 尝试解析JSON
            try:
                # 首先尝试直接解析
                result = json.loads(content)
            except json.JSONDecodeError:
                # 如果解析失败，尝试提取markdown代码块中的JSON
                try:
                    # 查找```json和```之间的内容（更宽松的匹配）
                    json_match = re.search(r'```json\s*\n?(.*?)\n?```', content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                        result = json.loads(json_content)
                    else:
                        # 尝试查找```和```之间的内容（没有json标记）
                        code_match = re.search(r'```\s*\n?(.*?)\n?```', content, re.DOTALL)
                        if code_match:
                            json_content = code_match.group(1).strip()
                            result = json.loads(json_content)
                        else:
                            print(f"   ❌ 无法找到JSON内容: {content}")
                            return []
                except json.JSONDecodeError as e:
                    print(f"   ❌ 无法解析GPT分类结果: {content}")
                    print(f"   🔍 JSON解析错误: {e}")
                    return []
            
            # 解析成功，提取结果
            diagram_figure_numbers = result.get("diagram_figures", [])
            reasoning = result.get("reasoning", "")
            
            print(f"   🤖 GPT分类结果: {reasoning}")
            
            # 根据GPT的分类结果筛选图片
            diagram_figures = []
            for i, fig in enumerate(figures, 1):
                if i in diagram_figure_numbers:
                    fig['type'] = 'diagram'
                    fig['gpt_reasoning'] = reasoning
                    diagram_figures.append(fig)
            
            return diagram_figures
                
        except Exception as e:
            print(f"   ❌ GPT分类失败: {str(e)}")
            return []
    
    def encode_image(self, image_path: str) -> str:
        """将图片编码为base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"   ❌ 图片编码失败: {e}")
            return ""
    
    def analyze_diagram_with_gpt4o(self, image_path: str, caption: str, context: str) -> Dict:
        """使用GPT-4o分析diagram图片内容"""
        
        if not self.setup_api_config():
            return {"error": "API配置失败"}
        
        # 编码图片
        base64_image = self.encode_image(image_path)
        if not base64_image:
            return {"error": "Failed to encode image"}
        
        # 使用prompt模板
        prompt = prompt_manager.get_diagram_analysis_prompt(caption, context)

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 3000,
            "temperature": 0.1
        }
        
        try:
            data = self.make_api_request_with_retry(payload, max_retries=3, delay=3.0)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # 尝试解析JSON
            try:
                # 首先尝试直接解析
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # 如果解析失败，尝试提取markdown代码块中的JSON
                try:
                    # 查找```json和```之间的内容（更宽松的匹配）
                    json_match = re.search(r'```json\s*\n?(.*?)\n?```', content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                        result = json.loads(json_content)
                        return result
                    else:
                        # 尝试查找```和```之间的内容（没有json标记）
                        code_match = re.search(r'```\s*\n?(.*?)\n?```', content, re.DOTALL)
                        if code_match:
                            json_content = code_match.group(1).strip()
                            result = json.loads(json_content)
                            return result
                        else:
                            # 如果没有找到代码块，尝试清理内容后解析
                            cleaned_content = content.strip()
                            # 移除可能的markdown格式
                            cleaned_content = re.sub(r'^```.*?\n', '', cleaned_content, flags=re.DOTALL)
                            cleaned_content = re.sub(r'\n```.*?$', '', cleaned_content, flags=re.DOTALL)
                            result = json.loads(cleaned_content)
                            return result
                except json.JSONDecodeError as e:
                    # 如果还是解析失败，返回原始内容用于调试
                    print(f"   ⚠️ JSON解析失败，原始响应: {content[:200]}...")
                    print(f"   🔍 JSON解析错误: {e}")
                    return {
                        "raw_response": content,
                        "error": "Failed to parse JSON response",
                        "diagram_analysis_available": False
                    }
                
        except Exception as e:
            return {"error": f"API request failed: {str(e)}"}
    
    def switch_api_source(self, new_source: APISource):
        """切换API源"""
        print(f"🔄 切换API源: {self.api_source.value} -> {new_source.value}")
        self.api_source = new_source
        return self.setup_api_config()
    
    def test_api_connection(self) -> bool:
        """测试API连接"""
        print(f"🔍 测试 {self.api_source.value} API连接...")
        
        test_payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, this is a test message."
                }
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        try:
            url = self.get_api_url()
            headers = self.get_api_headers()
            
            response = requests.post(url, headers=headers, json=test_payload, timeout=30)
            response.raise_for_status()
            
            print(f"✅ {self.api_source.value} API连接成功")
            return True
            
        except Exception as e:
            print(f"❌ {self.api_source.value} API连接失败: {str(e)}")
            return False


def extract_json_from_markdown(content: str) -> Dict:
    """从markdown内容中提取JSON"""
    try:
        # 查找```json和```之间的内容
        json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_content = json_match.group(1)
            return json.loads(json_content)
        
        # 尝试查找```和```之间的内容（没有json标记）
        code_match = re.search(r'```\s*\n(.*?)\n```', content, re.DOTALL)
        if code_match:
            json_content = code_match.group(1)
            return json.loads(json_content)
        
        # 如果都没有找到，尝试直接解析
        return json.loads(content)
        
    except json.JSONDecodeError:
        return {}


def generate_diagram_description_with_o3(caption: str, context: str) -> Dict:
    """使用GPT-o3同时生成长短两个版本的diagram描述"""
    
    # Azure OpenAI 配置
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = "o3-DR"  # 直接使用o3-DR部署
    api_version = "2025-01-01-preview"
    
    if not api_key:
        return {"error": "Azure OpenAI API key not found"}
    
    # 构建请求URL
    url = f"{endpoint}openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    
    # 使用prompt模板
    prompt = prompt_manager.get_diagram_description_prompt(caption, context)

    payload = {
        "model": "GPT4oVision",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_completion_tokens": 4000,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        return {
            "summary": content,
            "success": True
        }
            
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}


def generate_thinking_with_o3(caption: str, context: str, visual_analysis: str) -> Dict:
    """使用GPT-o3生成thinking描述，综合视觉分析和paper context"""
    
    # Azure OpenAI 配置
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = "o3-DR"  # 使用o3-DR部署
    api_version = "2025-01-01-preview"
    
    if not api_key:
        return {"error": "Azure OpenAI API key not found"}
    
    # 构建URL和headers
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    # 使用prompt_manager获取prompt
    prompt = prompt_manager.get_thinking_generation_prompt(caption, context, visual_analysis)
    
    payload = {
        "model": "GPT4oVision",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_completion_tokens": 4000,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        return {
            "summary": content,
            "success": True
        }
            
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}




def show_setup_help():
    """显示设置帮助信息"""
    print("\n🔧 设置帮助")
    print("=" * 50)
    print("🚫 重要更新: 我们强制使用SentenceTransformer本地模型避免Azure API限制")
    print()
    print("📦 必需依赖 (强制安装):")
    print("   pip install sentence-transformers torch")
    print("   或运行: python install_sentence_transformer.py")
    print()
    print("🔗 API配置 (用于GPT分析，不用于embedding):")
    print("1. Papyrus API (默认，推荐):")
    print("   export PAPYRUS_ENDPOINT='https://WestUS2Large.papyrus.binginternal.com/chat/completions'")
    print("   export PAPYRUS_VERIFY_SCOPE='api://5fe538a8-15d5-4a84-961e-be66cd036687/.default'")
    print("   export PAPYRUS_CLIENT_ID='d5702df1-96d9-4195-83a3-e44d8b0a0601'")
    print()
    print("2. 备选 Azure OpenAI API:")
    print("   export AZURE_OPENAI_ENDPOINT='https://your-endpoint.cognitiveservices.azure.com/'")
    print("   export AZURE_OPENAI_API_KEY='your-api-key-here'")
    print("   export AZURE_OPENAI_DEPLOYMENT='gpt-4o'")
    print()
    print("3. 对于Papyrus API，需要安装:")
    print("   pip install azure-identity")
    print()
    print("4. 如果Azure CLI token过期，重新登录:")
    print("   az login")
    print()
    print("💡 优势:")
    print("   ✅ 无API rate限制")
    print("   ✅ 本地处理，隐私安全")
    print("   ✅ 模型缓存，性能优化")
    print("   ✅ 离线可用")
    print()


def test_smart_markdown_paper(paper_dir: Path, paper_name: str, reasoner: HybridDiagramReasoner):
    """测试单个markdown论文的智能diagram分析"""
    print(f"\n{'='*60}")
    print(f"📚 测试论文: {paper_name}")
    print(f"{'='*60}")
    
    # 查找markdown文件
    markdown_path = paper_dir / "vlm" / f"{paper_name}.md"
    if not markdown_path.exists():
        print(f"❌ 未找到markdown文件: {markdown_path}")
        return None
    
    print(f"📄 读取markdown文件: {markdown_path}")
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None
    
    # 提取所有图片
    print("\n🔍 提取所有图片...")
    all_figures = reasoner.extract_all_figures_from_markdown(markdown_content)
    
    if not all_figures:
        print("❌ 未找到任何图片")
        return {
            "paper_name": paper_name,
            "paper_dir": str(paper_dir),
            "total_figures": 0,
            "diagram_figures": 0,
            "diagram_figures_list": [],
            "results": []
        }
    
    print(f"✅ 找到 {len(all_figures)} 个图片")
    
    # 使用GPT智能分类图片
    print("\n🤖 使用GPT智能分类图片...")
    diagram_figures = reasoner.classify_figures_with_gpt(all_figures, paper_name)
    
    if not diagram_figures:
        print("❌ GPT未识别出任何diagram图片")
        return {
            "paper_name": paper_name,
            "paper_dir": str(paper_dir),
            "total_figures": len(all_figures),
            "diagram_figures": 0,
            "diagram_figures_list": [],
            "results": []
        }
    
    print(f"✅ GPT识别出 {len(diagram_figures)} 个diagram图片")
    
    # 分析每个diagram图片
    results = []
    for i, figure in enumerate(diagram_figures, 1):
        print(f"\n📊 分析diagram图片 {i}: {figure['id']}")
        print(f"   Caption: {figure['caption'][:100]}...")
        
        # 构建图片路径
        image_path = paper_dir / "vlm" / figure['src']
        
        # 检查图片文件是否存在
        if not image_path.exists():
            print(f"   ❌ 图片文件不存在: {image_path}")
            # 尝试其他可能的路径
            possible_paths = [
                paper_dir / "vlm" / "images" / figure['src'],
                paper_dir / "vlm" / figure['src'].replace('images/', ''),
                paper_dir / figure['src'],
                paper_dir / "images" / figure['src']
            ]
            
            found = False
            for possible_path in possible_paths:
                if possible_path.exists():
                    image_path = possible_path
                    found = True
                    print(f"   ✅ 找到图片文件: {image_path}")
                    break
            
            if not found:
                print(f"   ❌ 所有可能的图片路径都不存在")
                continue
        
        # 使用检索提取相关上下文
        full_context = get_semantic_context_for_figure(markdown_content, figure['caption'])
        
        # 输出提取到的上下文用于调试
        print(f"   📄 提取到的上下文:")
        print(f"   {'='*50}")
        print(f"   {full_context[:300]}...")
        print(f"   {'='*50}")
        
        # 使用GPT-4o分析图片
        print("   🔍 使用GPT-4o分析图片...")
        diagram_analysis = reasoner.analyze_diagram_with_gpt4o(str(image_path), figure['caption'], full_context)
        
        if "error" in diagram_analysis:
            print(f"   ❌ 图片分析失败: {diagram_analysis['error']}")
            # 如果只是JSON解析失败，但仍然有原始响应，可以继续处理
            if diagram_analysis.get("raw_response"):
                print(f"   ⚠️ 但有原始响应可用，继续处理...")
            else:
                continue
        
        print("   ✅ 图片分析完成")
        
        # 使用GPT-o3生成绘图指令
        print("   🔍 使用GPT-o3生成绘图指令...")
        summary_result = generate_diagram_description_with_o3(
            figure['caption'], 
            full_context
        )
        
        if "error" in summary_result:
            print(f"   ❌ 绘图指令生成失败: {summary_result['error']}")
            # 即使GPT-o3失败，也保存GPT-4o的分析结果
            summary_result = {"error": "GPT-o3 summary generation failed", "diagram_analysis_available": True}
        
        print("   ✅ 绘图指令生成完成")
        
        # 解析GPT-o3的JSON响应
        parsed_summary = {}
        if "summary" in summary_result:
            try:
                parsed_summary = json.loads(summary_result["summary"])
                print(f"   ✅ JSON解析成功，包含字段: {list(parsed_summary.keys())}")
            except json.JSONDecodeError as e:
                print(f"   ⚠️ JSON解析失败: {e}")
                # 尝试从markdown代码块中提取
                parsed_summary = extract_json_from_markdown(summary_result["summary"])
                print(f"   📋 提取结果: {parsed_summary}")
        
        # 构建双阶段训练数据
        # 第一阶段输出 = 第二阶段输入 (diagram description)
        analysis = diagram_analysis.get("diagram_analysis", {})
        nodes = analysis.get("nodes", [])
        relationships = analysis.get("relationships", [])
        
        # 从nodes中提取main_components（使用新的nodes结构）
        main_components = []
        for node in nodes:
            if isinstance(node, dict):
                # 优先使用content，如果没有则使用id
                content = node.get("content", node.get("id", ""))
                if content:
                    main_components.append(content)
            else:
                # 向后兼容：如果是字符串，直接添加
                main_components.append(str(node))
        
        diagram_description = {
            "diagram_description": parsed_summary.get("diagram_description_long", ""),
            "diagram_description_long": parsed_summary.get("diagram_description_long", ""),
            "diagram_description_short": parsed_summary.get("diagram_description_short", ""),
            "diagram_type": analysis.get("diagram_type", ""),
            "main_components": main_components,
            "relationships": relationships
        }
        
        # 检查数据质量
        data_quality = "valid"
        quality_issues = []
        
        # 检查stage1数据质量
        if not diagram_description.get("diagram_description", "").strip():
            data_quality = "invalid"
            quality_issues.append("empty_diagram_description")
        
        # 检查stage2数据质量
        if "error" in summary_result:
            data_quality = "invalid"
            quality_issues.append("stage2_generation_failed")
        
        # 从GPT-o3结果中获取长短两个版本的描述
        diagram_desc_long = diagram_description.get("diagram_description_long", diagram_description.get("diagram_description", ""))
        diagram_desc_short = diagram_description.get("diagram_description_short", "")
        
        training_data = {
            # 数据质量标签
            "data_quality": data_quality,
            "quality_issues": quality_issues,
            
            # 第一阶段训练数据: context + caption -> diagram description (GPT-o3)
            "stage1_input": {
                "context": full_context,
                "caption": figure['caption']
            },
            
            # 第二阶段训练数据: diagram description -> thinking + image (GPT-4o)
            "stage2_input": {
                "diagram_description_long": diagram_desc_long,  # 长版本描述
                "diagram_description_short": diagram_desc_short  # 短版本描述（通过GPT生成）
            },
            "stage2_output": {
                "thinking_long": "",
                "thinking_short": "",
                "image_path": str(image_path)  # 第二阶段输出包含图片路径
            }
        }
        
        # 使用GPT-o3生成thinking，综合视觉分析和paper context
        print(f"   🔍 使用GPT-o3生成thinking...")
        thinking_result = generate_thinking_with_o3(
            figure['caption'], 
            full_context, 
            json.dumps(diagram_analysis, ensure_ascii=False)
        )
        
        judge_data_entry = None
        if "summary" in thinking_result:
            thinking_content = thinking_result["summary"]
            
            # 解析GPT-o3的JSON响应
            try:
                thinking_json = json.loads(thinking_content)
                thinking_long = thinking_json.get("thinking_long", "")
                thinking_short = thinking_json.get("thinking_short", "")
                print(f"   ✅ Thinking生成成功")
            except json.JSONDecodeError as e:
                print(f"   ⚠️ Thinking JSON解析失败: {e}")
                # 尝试从markdown代码块中提取
                thinking_json = extract_json_from_markdown(thinking_content)
                thinking_long = thinking_json.get("thinking_long", "")
                thinking_short = thinking_json.get("thinking_short", "")
            
            # 检查thinking是否为空或包含省略号
            if not thinking_long.strip() or "..." in thinking_long:
                data_quality = "invalid"
                quality_issues.append("empty_or_incomplete_thinking_long")
            
            if not thinking_short.strip() or "..." in thinking_short:
                data_quality = "invalid"
                quality_issues.append("empty_or_incomplete_thinking_short")
            
            training_data["stage2_output"]["thinking_long"] = thinking_long
            training_data["stage2_output"]["thinking_short"] = thinking_short
            
            # 更新数据质量标签
            training_data["data_quality"] = data_quality
            training_data["quality_issues"] = quality_issues
            
            # 为judge_data准备评分标准(rubric)数据
            analysis = diagram_analysis.get("diagram_analysis", {})
            nodes = analysis.get("nodes", [])
            groups = analysis.get("groups", [])
            relationships = analysis.get("relationships", [])
            visual_elements = analysis.get("visual_elements", {})
            standalone_nodes = analysis.get("standalone_nodes", [])
            
            # 提取所有节点（使用新的nodes结构）
            all_nodes = []
            
            # 从nodes数组中提取节点信息
            for node in nodes:
                if isinstance(node, dict):
                    # 使用新的nodes结构：包含id, type, content, attributes
                    node_info = {
                        "id": node.get("id", ""),
                        "type": node.get("type", ""),
                        "content": node.get("content", ""),
                        "attributes": node.get("attributes", {})
                    }
                    all_nodes.append(node_info)
                else:
                    # 向后兼容：如果是字符串，直接添加
                    all_nodes.append(node)
            
            # 添加独立节点（如果有的话）
            for standalone_node in standalone_nodes:
                if isinstance(standalone_node, str):
                    all_nodes.append(standalone_node)
                else:
                    all_nodes.append(standalone_node)
            
            # 从groups中提取节点引用（保持引用关系）
            group_node_refs = []
            for group in groups:
                group_node_refs.extend(group.get("nodes", []))
            
            judge_data_entry = {
                "image_info": {
                    "image_path": str(image_path),
                    "figure_id": figure['id'],
                    "figure_src": figure['src'],
                    "figure_caption": figure['caption']
                },
                "evaluation_rubric": {
                    "semantic_criteria": {
                        "critical_entities": all_nodes,
                        "critical_relationships": relationships,
                        "hierarchical_groups": groups,  # 直接使用新的groups结构
                        "data_flow": "Sequential processing flow",
                        "dependencies": []
                    },
                    "visual_criteria": {
                        "layout_requirements": visual_elements.get("layout", ""),
                        "color_scheme": visual_elements.get("colors", []),
                        "shape_requirements": visual_elements.get("shapes", [])
                    }
                },
                "reference_descriptions": {
                    "detailed_thinking": thinking_long,
                    "concise_thinking": thinking_short
                }
            }
        
        # 构建judge数据
        judge_data = judge_data_entry if judge_data_entry else {
            "image_info": {
                "image_path": str(image_path),
                "figure_id": figure['id'],
                "figure_src": figure['src'],
                "figure_caption": figure['caption']
            },
            "evaluation_rubric": {
                "semantic_criteria": {
                    "critical_entities": diagram_analysis.get("diagram_analysis", {}).get("nodes", []),
                    "critical_relationships": diagram_analysis.get("diagram_analysis", {}).get("relationships", []),
                    "hierarchical_groups": diagram_analysis.get("diagram_analysis", {}).get("groups", []),
                    "data_flow": "Sequential processing flow",
                    "dependencies": []
                },
                "visual_criteria": {
                    "layout_requirements": "",
                    "color_scheme": [],
                    "shape_requirements": []
                }
            },
            "reference_descriptions": {
                "detailed_thinking": "",
                "concise_thinking": ""
            }
        }
        
        result = {
            "training_data": training_data,
            "judge_data": judge_data
        }
        results.append(result)
    
    return {
        "paper_name": paper_name,
        "paper_dir": str(paper_dir),
        "total_figures": len(all_figures),
        "diagram_figures": len(diagram_figures),
        "diagram_figures_list": diagram_figures,
        "results": results
    }


def get_semantic_context_for_figure(markdown_content: str, caption: str) -> str:
    """使用FAISS检索找到与图片相关的上下文内容"""
    if not HAS_LANGCHAIN:
        print("   ❌ langchain 未安装，无法使用FAISS检索")
        return ""
    
    lines = markdown_content.split('\n')
    
    # 找到附录开始的位置（排除附录内容）
    appendix_start_idx = len(lines)  # 默认没有附录
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        # 检查是否是附录标题
        if (line_lower.startswith('# appendix') or 
            line_lower.startswith('## appendix') or
            line_lower.startswith('### appendix') or
            line_lower.startswith('# supplementary') or
            line_lower.startswith('## supplementary') or
            line_lower.startswith('### supplementary') or
            'appendix' in line_lower and line_lower.startswith('#')):
            appendix_start_idx = i
            break
    
    # 将文档分割成段落（只考虑正文部分，排除附录）
    paragraphs = []
    current_paragraph = []
    
    for i, line in enumerate(lines[:appendix_start_idx]):  # 只处理正文部分
        line = line.strip()
        if not line:
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        elif not line.startswith('![') and not line.startswith('#'):
            current_paragraph.append(line)
    
    # 添加最后一个段落
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # 过滤掉太短的段落
    paragraphs = [text for text in paragraphs if len(text) > 50]
    
    if not paragraphs:
        return ""
    
    # 使用FAISS检索
    try:
        print("   🔍 使用FAISS密集检索...")
        retriever = FAISSRetriever()
        retriever.fit(paragraphs)
        
        # 构建查询：使用caption作为查询，扩展相关词汇
        query = caption
        if any(keyword in caption.lower() for keyword in ['diagram', 'architecture', 'framework', 'pipeline', 'overview', 'structure']):
            query += " system design components modules workflow process"
        print(f"   🔍 FAISS查询: {query[:100]}...")
        
        # 搜索最相关的段落
        results = retriever.search(query, top_k=5)
        
        # 提取相关段落
        relevant_contexts = []
        for doc_idx, score in results:
            if score > 0:  # 只保留有分数的结果
                text = paragraphs[doc_idx]
                relevant_contexts.append(text)
                print(f"   📄 找到相关段落 (分数: {score:.3f}): {text[:100]}...")
        
        return '\n\n'.join(relevant_contexts)
        
    except Exception as e:
        print(f"   ❌ FAISS检索失败: {e}")
        return ""


class FAISSRetriever:
    """基于FAISS的密集向量检索器"""
    
    # 类级别的模型缓存，所有实例共享同一个模型
    _shared_model = None
    _shared_embeddings = None
    
    def __init__(self):
        self.vector_store = None
        self.documents = []
        self.embeddings = None
        
    def _get_embeddings(self):
        """获取embedding模型 - 强制使用SentenceTransformer避免API限制"""
        if self.embeddings is not None:
            return self.embeddings
        
        # 强制使用SentenceTransformer本地模型（避免Azure API rate限制）
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "❌ SentenceTransformer 未安装！\n"
                "请安装: pip install sentence-transformers torch\n"
                "我们强制使用本地模型以避免Azure API限制。"
            )
        
        try:
            # 检查是否已有共享的模型
            if FAISSRetriever._shared_embeddings is not None:
                self.embeddings = FAISSRetriever._shared_embeddings
                print(f"   🔗 复用共享的 SentenceTransformer Embeddings")
                return self.embeddings
            
            # 首次加载模型
            print("   🚀 强制使用 SentenceTransformer 本地模型...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"   📱 检测到设备: {device}")
            
            model = SentenceTransformer('all-mpnet-base-v2', device=device)
            print(f"   ✅ SentenceTransformer 模型加载成功")
            
            # 创建自定义的embeddings类来适配langchain
            try:
                from langchain.embeddings.base import Embeddings
            except ImportError:
                from langchain_core.embeddings import Embeddings
            
            class SentenceTransformerEmbeddings(Embeddings):
                    def __init__(self, model):
                        self.model = model
                    
                    def embed_documents(self, texts):
                        """嵌入文档列表"""
                        return self.model.encode(texts).tolist()
                    
                    def embed_query(self, text):
                        """嵌入单个查询"""
                        return self.model.encode([text])[0].tolist()
            
            # 创建embeddings实例并缓存到类级别
            FAISSRetriever._shared_model = model
            FAISSRetriever._shared_embeddings = SentenceTransformerEmbeddings(model)
            self.embeddings = FAISSRetriever._shared_embeddings
            
            print(f"   🔗 首次加载 SentenceTransformer Embeddings (设备: {device})")
            print(f"   💾 模型已缓存，后续实例将复用此模型")
            print(f"   🚫 已禁用Azure API，避免rate限制问题")
            return self.embeddings
            
        except Exception as e:
            error_msg = (
                f"❌ SentenceTransformer 初始化失败: {e}\n"
                f"🔧 解决方案:\n"
                f"   1. 确保已安装: pip install sentence-transformers torch\n"
                f"   2. 检查网络连接（首次使用需要下载模型）\n"
                f"   3. 确保有足够的磁盘空间\n"
                f"   4. 如果使用GPU，确保CUDA环境正确配置\n"
                f"🚫 我们不再使用Azure API以避免rate限制问题"
            )
            print(error_msg)
            raise RuntimeError(error_msg)
    
    def fit(self, documents: List[str]):
        """构建FAISS索引"""
        if not HAS_LANGCHAIN:
            raise ImportError("需要安装langchain来使用FAISS检索")
        
        self.documents = documents
        
        # 创建Document对象
        docs = [Document(page_content=doc) for doc in documents]
        
        # 文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        
        # 获取embeddings并构建FAISS索引
        embeddings = self._get_embeddings()
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        
        print(f"   📊 FAISS索引构建完成: {len(chunks)} 个chunks")
    
    def search(self, query: str, top_k: int = 3) -> List[tuple]:
        """搜索最相关的文档"""
        if self.vector_store is None:
            return []
        
        # 使用FAISS检索
        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        try:
            # 使用新的invoke方法
            docs = retriever.invoke(query)
        except AttributeError:
            # 回退到旧方法
            docs = retriever.get_relevant_documents(query)
        
        # 返回结果
        results = []
        for i, doc in enumerate(docs):
            # 找到原始文档的索引
            original_idx = self._find_original_doc_index(doc.page_content)
            if original_idx != -1:
                results.append((original_idx, 1.0 - i * 0.1))  # 简单的分数计算
        
        return results
    
    def _find_original_doc_index(self, content: str) -> int:
        """找到chunk对应的原始文档索引"""
        for i, doc in enumerate(self.documents):
            if content[:100] in doc:  # 使用前100个字符匹配
                return i
        return -1


def main():
    """主函数 - 测试所有markdown论文的智能diagram分析"""
    import argparse
    
    parser = argparse.ArgumentParser(description="混合图表分析器 - 支持多API源")
    parser.add_argument("--dedupe", action="store_true", default=True,
                        help="按arxiv ID去重，只处理每个论文的最新版本 (默认: True)")
    parser.add_argument("--no-dedupe", action="store_true", 
                        help="不去重，处理所有版本")
    parser.add_argument("--paper-id", type=str, 
                        help="只处理指定的arxiv ID (例如: 1905.12185)")
    
    args = parser.parse_args()
    
    # 处理去重参数
    dedupe = args.dedupe and not args.no_dedupe
    
    print("🤖 混合图表分析器 - 支持多API源")
    print("=" * 60)
    print(f"📚 去重模式: {'启用 (选择最新版本)' if dedupe else '禁用'}")
    if args.paper_id:
        print(f"🎯 指定论文ID: {args.paper_id}")
    
    # 检查是否有API配置
    has_azure_key = bool(os.getenv("AZURE_OPENAI_API_KEY"))
    has_azure_identity = HAS_AZURE_IDENTITY
    
    if not has_azure_key and not has_azure_identity:
        print("❌ 没有找到任何API配置")
        show_setup_help()
        return
    
    # 创建分析器实例
    reasoner = HybridDiagramReasoner(APISource.PAPYRUS)
    
    # 测试API连接
    if not reasoner.test_api_connection():
        print("❌ API连接失败")
        return
    
    # 论文目录
    papers_dir = Path(__file__).parent.parent.parent / "workspace" / "papers_markdown"
    
    # 获取所有论文目录
    all_paper_dirs = [d for d in papers_dir.iterdir() if d.is_dir()]
    
    if not all_paper_dirs:
        print("❌ 未找到论文目录")
        return
    
    # 处理论文选择逻辑
    if args.paper_id:
        # 只处理指定的论文ID
        paper_dirs = []
        for paper_dir in all_paper_dirs:
            paper_name = paper_dir.name
            import re
            match = re.match(r'(\d{4}\.\d{4,5})', paper_name)
            if match and match.group(1) == args.paper_id:
                paper_dirs.append(paper_dir)
        
        if not paper_dirs:
            print(f"❌ 未找到指定的论文ID: {args.paper_id}")
            return
        
        print(f"📚 找到 {len(paper_dirs)} 个匹配的论文版本")
        
    elif dedupe:
        # 按arxiv ID去重，只保留每个论文的第一个版本
        paper_groups = {}
        for paper_dir in all_paper_dirs:
            paper_name = paper_dir.name
            # 提取arxiv ID (例如: 1905.12185v3 -> 1905.12185)
            import re
            match = re.match(r'(\d{4}\.\d{4,5})', paper_name)
            if match:
                arxiv_id = match.group(1)
                if arxiv_id not in paper_groups:
                    paper_groups[arxiv_id] = []
                paper_groups[arxiv_id].append(paper_dir)
        
        # 每个arxiv ID只选择最后一个版本（最新版本）
        paper_dirs = []
        for arxiv_id, versions in paper_groups.items():
            # 按版本排序，选择最后一个（最新版本）
            versions.sort()
            selected_version = versions[-1]
            paper_dirs.append(selected_version)
            if len(versions) > 1:
                print(f"📚 {arxiv_id}: 找到 {len(versions)} 个版本，选择最新版本 {selected_version.name}")
        
        print(f"📚 去重后找到 {len(paper_dirs)} 个唯一论文")
        
    else:
        # 不去重，处理所有版本
        paper_dirs = all_paper_dirs
        print(f"📚 处理所有 {len(paper_dirs)} 个论文版本")
    
    # 测试每个论文
    all_results = []
    for paper_dir in paper_dirs:
        paper_name = paper_dir.name
        result = test_smart_markdown_paper(paper_dir, paper_name, reasoner)
        if result:
            all_results.append(result)
    
    # 分离训练数据和judge数据
    training_data = []
    judge_data = []
    
    for result in all_results:
        for item in result["results"]:
            if item.get("training_data"):
                training_data.append(item["training_data"])
            if item.get("judge_data"):
                judge_data.append(item["judge_data"])
    
    # 保存训练数据
    training_output_path = Path(__file__).parent / "hybrid_diagram_training_data.json"
    with open(training_output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # 保存judge数据
    judge_output_path = Path(__file__).parent / "hybrid_diagram_judge_data.json"
    with open(judge_output_path, 'w', encoding='utf-8') as f:
        json.dump(judge_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 训练数据已保存到: {training_output_path}")
    print(f"💾 Judge数据已保存到: {judge_output_path}")
    
    # 统计结果
    print(f"\n{'='*60}")
    print("📊 混合分析器结果统计")
    print(f"{'='*60}")
    
    total_papers = len(all_results)
    total_figures = sum(r["total_figures"] for r in all_results)
    total_diagrams = sum(r["diagram_figures"] for r in all_results)
    successful_analyses = 0
    valid_training_data = 0
    invalid_training_data = 0
    
    for result in all_results:
        paper_name = result["paper_name"]
        total_figs = result["total_figures"]
        diagram_figs = result["diagram_figures"]
        successful = len(result["results"])
        
        # 统计数据质量
        valid_count = sum(1 for r in result["results"] if r.get("training_data", {}).get("data_quality") == "valid")
        invalid_count = sum(1 for r in result["results"] if r.get("training_data", {}).get("data_quality") == "invalid")
        
        print(f"📚 {paper_name}:")
        print(f"   📊 总图片数: {total_figs}")
        print(f"   🤖 GPT识别diagram: {diagram_figs}")
        print(f"   ✅ 成功分析: {successful}/{diagram_figs}")
        print(f"   ✅ 有效训练数据: {valid_count}")
        print(f"   ❌ 无效训练数据: {invalid_count}")
        print()
        
        successful_analyses += successful
        valid_training_data += valid_count
        invalid_training_data += invalid_count
    
    print(f"🎯 总体统计:")
    print(f"   📚 测试论文数: {total_papers}")
    print(f"   📊 总图片数: {total_figures}")
    print(f"   🤖 GPT识别diagram: {total_diagrams}")
    print(f"   ✅ 成功分析: {successful_analyses}/{total_diagrams}")
    print(f"   📈 成功率: {successful_analyses/total_diagrams*100:.1f}%" if total_diagrams > 0 else "   📈 成功率: N/A")
    print(f"   🎯 识别率: {total_diagrams/total_figures*100:.1f}%" if total_figures > 0 else "   🎯 识别率: N/A")
    print(f"   ✅ 有效训练数据: {valid_training_data}")
    print(f"   ❌ 无效训练数据: {invalid_training_data}")
    print(f"   📊 数据质量率: {valid_training_data/(valid_training_data+invalid_training_data)*100:.1f}%" if (valid_training_data+invalid_training_data) > 0 else "   📊 数据质量率: N/A")
    
    print(f"\n💾 完整训练数据已保存到: {training_output_path}")
    print(f"✅ 最终使用的API源: {reasoner.api_source.value}")


if __name__ == "__main__":
    import json
    main()
