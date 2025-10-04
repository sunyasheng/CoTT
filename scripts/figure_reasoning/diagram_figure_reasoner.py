#!/usr/bin/env python3
"""
Diagram Figure Reasoner - 根据论文内容和图片分析生成绘图指令

功能：
1. 读取markdown文档
2. 提取diagram图片的caption和上下文
3. 使用GPT-4o分析图片内容
4. 生成绘图指令
5. 使用GPT-o3总结成特定格式
"""

import os
import re
import json
import requests
import base64
from pathlib import Path
from typing import Dict, List

# FAISS dense retrieval imports
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
    from langchain.schema import Document
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    print("⚠️ langchain 未安装，无法使用FAISS检索")

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

# 加载环境变量
load_env_vars()

def extract_diagram_figures(markdown_content: str) -> List[Dict]:
    """从markdown内容中提取diagram图片信息（排除附录中的图片）"""
    figures = []
    
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
    
    # 查找所有图片引用
    figure_pattern = r'<figure[^>]*id="([^"]*)"[^>]*>.*?<embed[^>]*src="([^"]*)"[^>]*/>.*?<figcaption>(.*?)</figcaption>.*?</figure>'
    matches = re.findall(figure_pattern, main_content, re.DOTALL)
    
    for match in matches:
        fig_id, src, caption = match
        # 清理caption中的HTML标签和多余空白
        caption = re.sub(r'<[^>]+>', '', caption).strip()
        caption = re.sub(r'\s+', ' ', caption)
        
        # 检查是否是diagram相关的图片，并排除附录中的图片
        if (('diagram' in fig_id.lower() or 'diagram' in caption.lower() or
             'architecture' in fig_id.lower() or 'architecture' in caption.lower() or
             'framework' in fig_id.lower() or 'framework' in caption.lower() or
             'pipeline' in fig_id.lower() or 'pipeline' in caption.lower() or
             'overview' in fig_id.lower() or 'overview' in caption.lower() or
             'structure' in fig_id.lower() or 'structure' in caption.lower()) and 
            'appendix_figures' not in src):
            figures.append({
                'id': fig_id,
                'src': src,
                'caption': caption,
                'type': 'diagram'
            })
    
    return figures


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
        elif not line.startswith('<figure') and not line.startswith('<embed') and not line.startswith('<figcaption'):
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
    
    # 构建提示词
    prompt = f"""You are analyzing a diagram from a research paper. Based on the image, caption, and context, provide detailed instructions for recreating this diagram.

**Figure Caption:**
{caption}

**Context around the figure:**
{context}

**Your task:**
Analyze the diagram and provide step-by-step instructions for drawing/recreating it. Focus on:
1. Overall structure and layout
2. Key components and their relationships
3. Visual elements (boxes, arrows, labels, etc.)
4. Color schemes and styling
5. Specific positioning and sizing

Please provide your analysis in the following JSON format:
{{
  "diagram_analysis": {{
    "diagram_type": "Type of diagram (e.g., architecture, flowchart, pipeline, etc.)",
    "main_components": ["List of main components/modules"],
    "relationships": ["List of key relationships between components"],
    "visual_elements": ["List of visual elements like boxes, arrows, labels"],
    "layout_structure": "Description of overall layout and organization"
  }},
  "drawing_instructions": {{
    "overall_approach": "General strategy for drawing this diagram",
    "step_by_step": [
      "Step 1: description",
      "Step 2: description",
      "..."
    ],
    "visual_specifications": {{
      "colors": ["List of colors used"],
      "shapes": ["List of shapes used"],
      "arrows": ["List of arrow types and directions"],
      "labels": ["List of labels and text elements"]
    }},
    "layout_guidelines": "Specific instructions for positioning and sizing",
    "styling_notes": "Additional styling and formatting notes"
  }},
  "detailed_description": "Detailed description of what the diagram shows and how it should be drawn"
}}"""

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
        response = requests.post(url, headers=headers, json=payload, timeout=120)
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
                    # 如果没有找到代码块，返回原始内容
                    return {
                        "raw_response": content,
                        "error": "Failed to parse JSON response"
                    }
            except json.JSONDecodeError:
                # 如果还是解析失败，返回原始内容
                return {
                    "raw_response": content,
                    "error": "Failed to parse JSON response"
                }
            
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}


def generate_drawing_summary_with_o3(caption: str, context: str, diagram_analysis: Dict, paper_content: str) -> Dict:
    """使用GPT-o3生成适合AI绘图工具的简洁指令"""
    
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
    
    # 构建提示词
    prompt = f"""You are analyzing a research paper diagram and need to generate a concise drawing instruction for AI image generation tools like DALL-E 3 or Nano Banana.

**Figure Caption:**
{caption}

**Context around the figure:**
{context}

**Diagram Analysis:**
{json.dumps(diagram_analysis, indent=2)}

**Your task:**
Generate a summary in the specific format requested. The output should be in this exact format:

<think>
This paper presents [brief description of the paper's main contribution]. The diagram shows [description of what the diagram illustrates]. The key components include [list main components]. The relationships between components are [describe relationships]. This diagram is important because [explain significance].
</think>

<tool>draw</tool>
<params>
[Generate a concise, clear instruction for AI image generation tools. The instruction should be:
1. Simple and direct (under 200 words)
2. Focus on visual elements that AI can understand and generate
3. Use clear, descriptive language
4. Include key components, layout, and visual style
5. Avoid complex technical jargon
6. Be specific about colors, shapes, and arrangement
7. Suitable for DALL-E 3, Midjourney, or similar AI art tools

Example format: "Create a clean, professional diagram showing [main concept]. The layout should be [layout description]. Include [key components] arranged [how they're arranged]. Use [color scheme] with [visual style]. The diagram should have [specific visual elements] and show [relationships between elements]."]
</params>

Please provide your response in the exact format above."""

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


def test_diagram_paper(paper_dir: Path, paper_name: str):
    """测试单个论文的diagram分析"""
    print(f"\n{'='*60}")
    print(f"📚 测试论文: {paper_name}")
    print(f"{'='*60}")
    
    # 查找markdown文件
    markdown_path = paper_dir / "output_pandoc.md"
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
    
    # 提取diagram图片
    print("\n🔍 提取diagram图片...")
    figures = extract_diagram_figures(markdown_content)
    
    if not figures:
        print("❌ 未找到diagram图片")
        return {
            "paper_name": paper_name,
            "paper_dir": str(paper_dir),
            "figures_found": 0,
            "figures": [],
            "results": []
        }
    
    print(f"✅ 找到 {len(figures)} 个diagram图片")
    
    # 分析每个图片
    results = []
    for i, figure in enumerate(figures, 1):
        print(f"\n📊 分析图片 {i}: {figure['id']}")
        print(f"   Caption: {figure['caption'][:100]}...")
        
        # 构建图片路径，将PDF后缀替换为PNG
        src_path = figure['src']
        if src_path.endswith('.pdf'):
            src_path = src_path.replace('.pdf', '.png')
        
        image_path = paper_dir / src_path
        if not image_path.exists():
            print(f"   ❌ 图片文件不存在: {image_path}")
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
            continue
        
        print("   ✅ 图片分析完成")
        
        # 使用GPT-o3生成总结
        print("   🔍 使用GPT-o3生成绘图指令...")
        summary_result = generate_drawing_summary_with_o3(
            figure['caption'], 
            full_context, 
            diagram_analysis,
            markdown_content
        )
        
        if "error" in summary_result:
            print(f"   ❌ 总结生成失败: {summary_result['error']}")
            # 即使GPT-o3失败，也保存GPT-4o的分析结果
            summary_result = {"error": "GPT-o3 summary generation failed", "diagram_analysis_available": True}
        
        print("   ✅ 绘图指令生成完成")
        
        result = {
            "figure_info": figure,
            "context": full_context,
            "diagram_analysis": diagram_analysis,
            "drawing_summary": summary_result
        }
        results.append(result)
        
        # 显示结果
        if "error" not in summary_result or summary_result.get("diagram_analysis_available"):
            print("   📋 生成的绘图指令:")
            print("   " + "="*50)
            if "summary" in summary_result:
                print(f"   {summary_result.get('summary', '')[:200]}...")
            else:
                print("   GPT-4o分析结果已保存，但GPT-o3总结生成失败")
            print("   " + "="*50)
    
    return {
        "paper_name": paper_name,
        "paper_dir": str(paper_dir),
        "figures_found": len(figures),
        "figures": figures,
        "results": results
    }


def main():
    """主函数 - 测试所有论文的diagram分析"""
    print("🔍 测试所有论文的diagram分析")
    print("=" * 60)
    
    # 论文目录
    papers_dir = Path(__file__).parent.parent.parent / "workspace" / "papers_latex"
    
    # 获取所有论文目录
    paper_dirs = [d for d in papers_dir.iterdir() if d.is_dir() and d.name.startswith('arXiv-')]
    
    if not paper_dirs:
        print("❌ 未找到论文目录")
        return
    
    print(f"📚 找到 {len(paper_dirs)} 个论文目录")
    
    # 测试每个论文
    all_results = []
    for paper_dir in paper_dirs:
        paper_name = paper_dir.name
        result = test_diagram_paper(paper_dir, paper_name)
        if result:
            all_results.append(result)
    
    # 保存所有结果
    output_path = Path(__file__).parent / "all_diagrams_test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 统计结果
    print(f"\n{'='*60}")
    print("📊 测试结果统计")
    print(f"{'='*60}")
    
    total_papers = len(all_results)
    total_figures = sum(r["figures_found"] for r in all_results)
    successful_analyses = 0
    
    for result in all_results:
        paper_name = result["paper_name"]
        figures_found = result["figures_found"]
        successful = sum(1 for r in result["results"] if "error" not in r.get("drawing_summary", {}))
        
        print(f"📚 {paper_name}:")
        print(f"   📊 找到 {figures_found} 个diagram图片")
        print(f"   ✅ 成功分析 {successful}/{figures_found} 个图片")
        
        successful_analyses += successful
    
    print(f"\n🎯 总体统计:")
    print(f"   📚 测试论文数: {total_papers}")
    print(f"   📊 总图片数: {total_figures}")
    print(f"   ✅ 成功分析: {successful_analyses}/{total_figures}")
    print(f"   📈 成功率: {successful_analyses/total_figures*100:.1f}%" if total_figures > 0 else "   📈 成功率: N/A")
    
    print(f"\n💾 详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
