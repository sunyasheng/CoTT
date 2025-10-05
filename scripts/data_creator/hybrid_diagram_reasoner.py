#!/usr/bin/env python3
"""
Hybrid Diagram Figure Reasoner - 支持多API源的智能图表分析器

功能：
1. 支持切换不同的GPT-4o接口源（Azure OpenAI 和 Papyrus）
2. 提取所有图片的caption
3. 使用GPT判断哪些是diagram
4. 只分析被GPT识别为diagram的图片
5. 生成绘图指令

支持的API源：
- Azure OpenAI (默认)
- Papyrus (Microsoft内部API)
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
    
    def __init__(self, api_source: APISource = APISource.AZURE_OPENAI):
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
        ]
        
        if HAS_DOTENV:
            for env_path in env_paths:
                if env_path.exists():
                    print(f"📄 加载环境变量: {env_path}")
                    load_dotenv(env_path)
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
                    
                    # 设置默认的 Azure OpenAI 配置
                    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
                        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com/"
                    if not os.getenv("AZURE_OPENAI_DEPLOYMENT"):
                        os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o"
                    if not os.getenv("AZURE_OPENAI_API_VERSION"):
                        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-15-preview"
                    print(f"   ✅ 设置默认 Azure OpenAI 配置")
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
            
            self.papyrus_endpoint = "https://WestUS2Large.papyrus.binginternal.com/chat/completions"
            self.verify_scope = "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default"
            self.client_id = "d5702df1-96d9-4195-83a3-e44d8b0a0601"
            
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
        try:
            # 尝试使用AzureCliCredential
            print("🔐 尝试使用 Azure CLI 认证...")
            cred = AzureCliCredential()
            self.access_token = cred.get_token(self.verify_scope).token
            print("✅ Azure CLI 认证成功")
            return True
        except Exception as e:
            print(f"❌ Azure CLI 认证失败: {e}")
        
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
            # 尝试使用ManagedIdentityCredential（仅在Azure环境中有效）
            print("🔐 尝试使用 Managed Identity 认证...")
            cred = ManagedIdentityCredential(client_id=self.client_id)
            self.access_token = cred.get_token(self.verify_scope).token
            print("✅ Managed Identity 认证成功")
            return True
        except Exception as e:
            print(f"❌ Managed Identity 认证失败: {e}")
        
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
                "papyrus-model-name": "gpt4ovision-batch",
                "papyrus-timeout-ms": "30000",
                "papyrus-quota-id": "msftaicopilot/windowsdata",
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


def test_api_switching():
    """测试API源切换功能"""
    print("🧪 测试API源切换功能")
    print("=" * 50)
    
    # 创建分析器实例
    reasoner = HybridDiagramReasoner(APISource.AZURE_OPENAI)
    
    # 测试Azure OpenAI连接
    print("\n1. 测试Azure OpenAI连接...")
    azure_success = reasoner.test_api_connection()
    
    if azure_success:
        print("✅ Azure OpenAI连接成功")
    else:
        print("❌ Azure OpenAI连接失败")
    
    # 尝试切换到Papyrus
    print("\n2. 尝试切换到Papyrus API...")
    papyrus_success = reasoner.switch_api_source(APISource.PAPYRUS)
    
    if papyrus_success:
        print("✅ 成功切换到Papyrus API")
        papyrus_connection = reasoner.test_api_connection()
        if papyrus_connection:
            print("✅ Papyrus API连接成功")
        else:
            print("❌ Papyrus API连接失败")
    else:
        print("❌ 无法切换到Papyrus API")
    
    # 切换回Azure OpenAI
    print("\n3. 切换回Azure OpenAI...")
    azure_switch = reasoner.switch_api_source(APISource.AZURE_OPENAI)
    
    if azure_switch:
        print("✅ 成功切换回Azure OpenAI")
        azure_connection = reasoner.test_api_connection()
        if azure_connection:
            print("✅ Azure OpenAI连接成功")
        else:
            print("❌ Azure OpenAI连接失败")
    else:
        print("❌ 无法切换回Azure OpenAI")
    
    print(f"\n🎯 最终API源: {reasoner.api_source.value}")
    return reasoner


def test_image_analysis(reasoner):
    """测试图片分析功能"""
    print("\n🧪 测试图片分析功能")
    print("=" * 50)
    
    # 查找测试图片
    test_image_path = Path(__file__).parent.parent.parent / "workspace" / "reference" / "math.png"
    
    if not test_image_path.exists():
        print(f"❌ 测试图片不存在: {test_image_path}")
        return
    
    print(f"📸 找到测试图片: {test_image_path}")
    
    # 测试图片分析
    print("🔍 开始分析图片...")
    result = reasoner.analyze_diagram_with_gpt4o(
        str(test_image_path),
        "A mathematical problem diagram",
        "This is a test context for the mathematical diagram."
    )
    
    if "error" in result:
        print(f"❌ 图片分析失败: {result['error']}")
    else:
        print("✅ 图片分析成功")
        print(f"📊 分析结果: {json.dumps(result, indent=2, ensure_ascii=False)[:200]}...")


def example_batch_processing(reasoner):
    """批量处理示例"""
    print("\n📦 批量处理示例")
    print("=" * 50)
    
    # 模拟markdown内容
    sample_markdown = """
# Sample Paper

This is a sample paper with some diagrams.

![Figure 1](images/diagram1.png)
Figure 1: System architecture diagram showing the main components and their relationships.

![Figure 2](images/chart1.png)
Figure 2: Performance comparison chart.

![Figure 3](images/flowchart.png)
Figure 3: Process flowchart illustrating the workflow.
"""
    
    # 提取所有图片
    print("🔍 提取图片信息...")
    figures = reasoner.extract_all_figures_from_markdown(sample_markdown)
    print(f"📊 找到 {len(figures)} 个图片")
    
    for i, fig in enumerate(figures, 1):
        print(f"   {i}. {fig['id']}: {fig['caption'][:50]}...")
    
    # 使用GPT分类图片
    print("\n🤖 使用GPT分类图片...")
    diagram_figures = reasoner.classify_figures_with_gpt(figures, "Sample Paper")
    print(f"📊 GPT识别出 {len(diagram_figures)} 个diagram图片")
    
    for fig in diagram_figures:
        print(f"   ✅ {fig['id']}: {fig['caption'][:50]}...")


def main():
    """主函数 - 演示混合API功能"""
    print("🤖 混合图表分析器 - 支持多API源")
    print("=" * 60)
    
    try:
        # 测试API切换
        reasoner = test_api_switching()
        
        # 测试图片分析
        test_image_analysis(reasoner)
        
        # 批量处理示例
        example_batch_processing(reasoner)
        
        print("\n🎉 所有测试完成")
        print(f"✅ 最终使用的API源: {reasoner.api_source.value}")
        
    except Exception as e:
        print(f"❌ 运行测试时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import json
    main()
