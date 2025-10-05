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
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from prompts_template.prompt_manager import prompt_manager
from langchain.schema import Document
HAS_LANGCHAIN = True

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
            url = self.get_api_url()
            headers = self.get_api_headers()
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # 尝试解析JSON
            try:
                # 首先尝试直接解析
                result = json.loads(content)
            except json.JSONDecodeError:
                # 如果解析失败，尝试提取markdown代码块中的JSON
                try:
                    # 查找```json和```之间的内容
                    json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
                        result = json.loads(json_content)
                    else:
                        # 尝试查找```和```之间的内容（没有json标记）
                        code_match = re.search(r'```\s*\n(.*?)\n```', content, re.DOTALL)
                        if code_match:
                            json_content = code_match.group(1)
                            result = json.loads(json_content)
                        else:
                            print(f"   ❌ 无法找到JSON内容: {content}")
                            return []
                except json.JSONDecodeError:
                    print(f"   ❌ 无法解析GPT分类结果: {content}")
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
            url = self.get_api_url()
            headers = self.get_api_headers()
            
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # 尝试解析JSON
            try:
                # 首先尝试直接解析
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # 如果解析失败，尝试提取markdown代码块中的JSON
                try:
                    # 查找```json和```之间的内容
                    json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
                        result = json.loads(json_content)
                        return result
                    else:
                        # 尝试查找```和```之间的内容（没有json标记）
                        code_match = re.search(r'```\s*\n(.*?)\n```', content, re.DOTALL)
                        if code_match:
                            json_content = code_match.group(1)
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
                except json.JSONDecodeError:
                    # 如果还是解析失败，返回原始内容用于调试
                    print(f"   ⚠️ JSON解析失败，原始响应: {content[:200]}...")
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




def show_setup_help():
    """显示设置帮助信息"""
    print("\n🔧 设置帮助")
    print("=" * 50)
    print("此工具默认使用 Papyrus API (Microsoft内部API)")
    print("要使用此工具，需要设置以下环境变量:")
    print()
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
        
        # 构建训练数据
        training_data = {
            "data_quality": "valid",
            "quality_issues": [],
            "stage1_input": {
                "context": full_context,
                "caption": figure['caption']
            },
            "stage2_input": {
                "diagram_description": diagram_analysis.get("diagram_analysis", {}).get("diagram_type", ""),
                "main_components": diagram_analysis.get("diagram_analysis", {}).get("nodes", [])
            },
            "stage2_output": {
                "thinking": "",
                "image_path": str(image_path)
            }
        }
        
        # 构建judge数据
        judge_data = {
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
    
    def __init__(self):
        self.vector_store = None
        self.documents = []
        self.embeddings = None
        
    def _get_embeddings(self):
        """获取embedding模型"""
        if self.embeddings is not None:
            return self.embeddings
            
        # 检查Azure OpenAI配置
        if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
            
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=endpoint,
                openai_api_version=api_version,
                openai_api_key=api_key,
                model="text-embedding-3-large"
            )
            print("   🔗 使用 Azure OpenAI Embeddings")
        else:
            # 使用OpenAI API
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            print("   🔗 使用 OpenAI Embeddings")
        
        return self.embeddings
    
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
    print("🤖 混合图表分析器 - 支持多API源")
    print("=" * 60)
    
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
    paper_dirs = [d for d in papers_dir.iterdir() if d.is_dir()]
    
    if not paper_dirs:
        print("❌ 未找到论文目录")
        return
    
    print(f"📚 找到 {len(paper_dirs)} 个论文目录")
    
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
