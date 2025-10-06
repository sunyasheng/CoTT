#!/usr/bin/env python3
"""
Smart Diagram Figure Reasoner - 使用GPT智能判断图片类型

功能：
1. 提取所有图片的caption
2. 使用GPT判断哪些是diagram
3. 只分析被GPT识别为diagram的图片
4. 生成绘图指令

这种方法比关键词匹配更准确和稳定
"""

import os
import re
import json
import requests
import base64
import time
from pathlib import Path
from typing import Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from prompts_template.prompt_manager import prompt_manager
from langchain.schema import Document
HAS_LANGCHAIN = True

from dotenv import load_dotenv
HAS_DOTENV = True

def load_env_vars():
    """加载环境变量"""
    # 尝试从多个位置加载 .env 文件
    env_paths = [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent.parent.parent / "CoTT" / ".env",
        Path(__file__).parent.parent.parent / "CoTT" / ".env_old",
        Path("/home/suny0a/Proj/CoTT/.env"),  # Linux 绝对路径
        Path("/Users/suny0a/Proj/MM-Reasoning/CoTT/.env"),  # macOS 绝对路径
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

# 加载环境变量
load_env_vars()

# 全局变量用于控制API请求间隔
last_api_call_time = 0
api_call_interval = 1.0  # 1秒间隔，避免rate limit

def make_api_request_with_retry(url: str, headers: Dict, payload: Dict, max_retries: int = 3, delay: float = 2.0, timeout: int = 180) -> Dict:
    """带重试机制的API请求 - 与hybrid版本保持一致"""
    global last_api_call_time, api_call_interval
    
    for attempt in range(max_retries):
        try:
            # 添加请求间隔，避免rate limit
            current_time = time.time()
            time_since_last_call = current_time - last_api_call_time
            if time_since_last_call < api_call_interval:
                sleep_time = api_call_interval - time_since_last_call
                print(f"   ⏳ 等待 {sleep_time:.1f} 秒避免rate limit...")
                time.sleep(sleep_time)
            
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            last_api_call_time = time.time()  # 更新最后调用时间
            
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
            elif response.status_code == 408:
                print(f"   ⚠️ 408 Request Timeout (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    wait_time = delay * (1.5 ** attempt)  # 适中的退避
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

def extract_all_figures_from_markdown(markdown_content: str) -> List[Dict]:
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


def classify_figures_with_gpt(figures: List[Dict], paper_title: str = "") -> List[Dict]:
    """使用GPT判断哪些图片是diagram"""
    
    # Azure OpenAI 配置
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = "gpt-4o"  # 使用gpt-4o进行分类
    api_version = "2024-02-15-preview"
    
    if not api_key:
        print("❌ Azure OpenAI API key not found")
        return []
    
    # 构建请求URL
    url = f"{endpoint}openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    
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
        data = make_api_request_with_retry(url, headers, payload, max_retries=3, delay=2.0, timeout=60)
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


def encode_image(image_path: str) -> str:
    """将图片编码为base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"   ❌ 图片编码失败: {e}")
        return ""


def analyze_diagram_with_gpt4o(image_path: str, caption: str, context: str) -> Dict:
    """使用GPT-4o分析diagram图片内容"""
    
    # Azure OpenAI 配置
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = "gpt-4o"  # 直接使用gpt-4o部署
    api_version = "2024-02-15-preview"
    
    if not api_key:
        return {"error": "Azure OpenAI API key not found"}
    
    # 构建请求URL
    url = f"{endpoint}openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    
    # 编码图片
    base64_image = encode_image(image_path)
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
        data = make_api_request_with_retry(url, headers, payload, max_retries=3, delay=3.0, timeout=180)
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
                # 如果还是解析失败，尝试从raw_response中提取（这是修复的关键）
                if "raw_response" in content:
                    # 如果content本身就是raw_response格式，尝试提取其中的JSON
                    raw_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
                    if raw_match:
                        json_content = raw_match.group(1)
                        result = json.loads(json_content)
                        return result
                
                # 如果还是解析失败，返回原始内容用于调试
                print(f"   ⚠️ JSON解析失败，原始响应: {content[:200]}...")
                return {
                    "raw_response": content,
                    "error": "Failed to parse JSON response",
                    "diagram_analysis_available": False
                }
            
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}


def extract_flow_from_relationships(relationships: list) -> str:
    """从关系中提取数据流描述"""
    if not relationships:
        return "Sequential processing flow"
    
    # 分析关系中的流向
    for rel in relationships:
        if isinstance(rel, dict):
            rel_text = rel.get("description", "")
            rel_type = rel.get("type", "")
        else:
            rel_text = str(rel)
            rel_type = ""
        
        if "->" in rel_text or "data_flow" in rel_type:
            return "Directed flow with clear input-output relationships"
        elif "parallel" in rel_text.lower() or "parallel" in rel_type.lower():
            return "Parallel processing with convergence"
    
    return "Complex interconnected flow"


def extract_dependencies_from_relationships(relationships: list) -> list:
    """从关系中提取依赖关系"""
    dependencies = []
    for rel in relationships:
        if isinstance(rel, dict):
            # 已经是字典格式，直接使用
            dependencies.append({
                "from": rel.get("from", ""),
                "to": rel.get("to", ""),
                "type": rel.get("type", "dependency")
            })
        else:
            # 字符串格式，解析
            rel_str = str(rel)
            if "->" in rel_str:
                parts = rel_str.split("->")
                if len(parts) == 2:
                    dependencies.append({
                        "from": parts[0].strip(),
                        "to": parts[1].strip(),
                        "type": "data_flow"
                    })
    return dependencies


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
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_completion_tokens": 4000
    }
    
    try:
        data = make_api_request_with_retry(url, headers, payload, max_retries=3, delay=2.0, timeout=120)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        return {
            "summary": content,
            "success": True
        }
            
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}


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
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_completion_tokens": 4000
    }
    
    try:
        data = make_api_request_with_retry(url, headers, payload, max_retries=3, delay=2.0, timeout=120)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        return {
            "summary": content,
            "success": True
        }
            
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}


def test_smart_markdown_paper(paper_dir: Path, paper_name: str):
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
    all_figures = extract_all_figures_from_markdown(markdown_content)
    
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
    diagram_figures = classify_figures_with_gpt(all_figures, paper_name)
    
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
        diagram_analysis = analyze_diagram_with_gpt4o(str(image_path), figure['caption'], full_context)
        
        if "error" in diagram_analysis:
            print(f"   ❌ 图片分析失败: {diagram_analysis['error']}")
            # 如果只是JSON解析失败，但仍然有原始响应，可以继续处理
            if diagram_analysis.get("raw_response"):
                print(f"   ⚠️ 但有原始响应可用，继续处理...")
            else:
                continue
        
        print("   ✅ 图片分析完成")
        
        # 使用GPT-o3生成总结
        print("   🔍 使用GPT-o3生成绘图指令...")
        summary_result = generate_diagram_description_with_o3(
            figure['caption'], 
            full_context
        )
        
        if "error" in summary_result:
            print(f"   ❌ 总结生成失败: {summary_result['error']}")
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
                        "data_flow": extract_flow_from_relationships(relationships),
                        "dependencies": extract_dependencies_from_relationships(relationships)
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
            
            # 不再提取drawing instructions，因为不需要step-by-step描述
        
        result = {
            "training_data": training_data,
            "judge_data": judge_data_entry
        }
        results.append(result)
        
        # 显示结果
        if "error" not in summary_result or summary_result.get("diagram_analysis_available"):
            print("   📋 生成的绘图指令:")
            print("   " + "="*50)
            if "summary" in summary_result:
                full_response = summary_result.get('summary', '')
                print(f"   {full_response[:200]}...")
                print(f"   📊 完整响应长度: {len(full_response)} 字符")
                # 检查是否包含两个字段
                if "diagram_description_short" in full_response:
                    print("   ✅ 包含短描述字段")
                else:
                    print("   ❌ 缺少短描述字段")
            else:
                print("   GPT-4o分析结果已保存，但GPT-o3总结生成失败")
            print("   " + "="*50)
    
    return {
        "paper_name": paper_name,
        "paper_dir": str(paper_dir),
        "total_figures": len(all_figures),
        "diagram_figures": len(diagram_figures),
        "diagram_figures_list": diagram_figures,  # 只保存有用的diagram图片
        "results": results
    }


def main():
    """主函数 - 测试所有markdown论文的智能diagram分析"""
    print("🤖 智能diagram分析 - 使用GPT判断图片类型")
    print("=" * 60)
    
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
        result = test_smart_markdown_paper(paper_dir, paper_name)
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
    training_output_path = Path(__file__).parent / "diagram_training_data.json"
    with open(training_output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # 保存judge数据
    judge_output_path = Path(__file__).parent / "diagram_judge_data.json"
    with open(judge_output_path, 'w', encoding='utf-8') as f:
        json.dump(judge_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 训练数据已保存到: {training_output_path}")
    print(f"💾 Judge数据已保存到: {judge_output_path}")
    
    # 统计结果
    print(f"\n{'='*60}")
    print("📊 智能分析结果统计")
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
        successful = sum(1 for r in result["results"] if "error" not in r.get("drawing_summary", {}))
        
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


if __name__ == "__main__":
    main()
