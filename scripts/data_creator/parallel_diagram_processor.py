#!/usr/bin/env python3
"""
并行图表处理器 - 多线程处理markdown文件

功能：
1. 扫描指定目录下的所有markdown文件
2. 使用多线程并行处理每个文件
3. 调用hybrid_diagram_reasoner.py进行图表分析
4. 合并所有结果到统一的输出文件

使用方法：
python parallel_diagram_processor.py --input_dir /path/to/markdown/files --output_dir /path/to/output --workers 4
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import time
from datetime import datetime

# 添加当前目录到Python路径，以便导入hybrid_diagram_reasoner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from hybrid_diagram_reasoner import HybridDiagramReasoner, APISource
except ImportError as e:
    print(f"❌ 无法导入hybrid_diagram_reasoner: {e}")
    print("请确保hybrid_diagram_reasoner.py在同一目录下")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parallel_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ParallelDiagramProcessor:
    """并行图表处理器"""
    
    def __init__(self, api_source: APISource = APISource.PAPYRUS, max_workers: int = 4):
        self.api_source = api_source
        self.max_workers = max_workers
        self.reasoner = None
        self.results = []
        self.failed_files = []
        
    def initialize_reasoner(self):
        """初始化reasoner"""
        try:
            self.reasoner = HybridDiagramReasoner(self.api_source)
            logger.info(f"✅ 初始化reasoner成功，使用API: {self.api_source.value}")
            return True
        except Exception as e:
            logger.error(f"❌ 初始化reasoner失败: {e}")
            return False
    
    def find_markdown_files(self, input_dir: str) -> List[Path]:
        """扫描目录下的所有markdown文件"""
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"❌ 输入目录不存在: {input_dir}")
            return []
        
        # 查找所有.md文件
        markdown_files = []
        for pattern in ["**/*.md", "**/*.markdown"]:
            markdown_files.extend(input_path.glob(pattern))
        
        # 过滤掉隐藏文件和临时文件
        markdown_files = [f for f in markdown_files if not f.name.startswith('.')]
        
        logger.info(f"📁 在 {input_dir} 中找到 {len(markdown_files)} 个markdown文件")
        return markdown_files
    
    def process_single_file(self, markdown_file: Path, output_dir: Path) -> Dict[str, Any]:
        """处理单个markdown文件"""
        start_time = time.time()
        file_result = {
            "file_path": str(markdown_file),
            "file_name": markdown_file.name,
            "status": "processing",
            "start_time": datetime.now().isoformat(),
            "error": None,
            "training_data": [],
            "judge_data": [],
            "statistics": {}
        }
        
        try:
            logger.info(f"🔄 开始处理: {markdown_file.name}")
            
            # 创建每个文件的输出目录
            file_output_dir = output_dir / markdown_file.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 调用hybrid_diagram_reasoner处理单个文件
            # 这里需要修改hybrid_diagram_reasoner.py以支持单文件处理
            result = self.process_markdown_file_with_reasoner(markdown_file, file_output_dir)
            
            if result:
                file_result.update(result)
                file_result["status"] = "completed"
                logger.info(f"✅ 完成处理: {markdown_file.name}")
            else:
                file_result["status"] = "failed"
                file_result["error"] = "处理返回空结果"
                logger.error(f"❌ 处理失败: {markdown_file.name}")
                
        except Exception as e:
            file_result["status"] = "failed"
            file_result["error"] = str(e)
            logger.error(f"❌ 处理文件时出错 {markdown_file.name}: {e}")
        
        file_result["end_time"] = datetime.now().isoformat()
        file_result["processing_time"] = time.time() - start_time
        
        return file_result
    
    def process_markdown_file_with_reasoner(self, markdown_file: Path, output_dir: Path) -> Dict[str, Any]:
        """使用reasoner处理单个markdown文件"""
        try:
            # 这里需要调用hybrid_diagram_reasoner中的单文件处理函数
            # 由于原代码是处理整个目录的，我们需要提取单文件处理逻辑
            
            # 读取markdown文件
            with open(markdown_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取图片信息
            figures = self.reasoner.extract_all_figures_from_markdown(content)
            
            if not figures:
                logger.warning(f"⚠️ 文件中没有找到图片: {markdown_file.name}")
                return {
                    "training_data": [],
                    "judge_data": [],
                    "statistics": {
                        "total_figures": 0,
                        "diagram_figures": 0,
                        "processed_figures": 0
                    }
                }
            
            # 使用GPT分类图片
            diagram_figures = self.reasoner.classify_figures_with_gpt(figures, content)
            
            training_data = []
            judge_data = []
            
            # 处理每个diagram图片
            for i, figure in enumerate(diagram_figures):
                try:
                    # 构建图片路径
                    # 从src中提取文件名
                    src_path = figure['src']
                    if '/' in src_path:
                        filename = src_path.split('/')[-1]
                    else:
                        filename = src_path
                    # markdown_file.parent 已经是 vlm 目录，所以直接加 images
                    image_path = markdown_file.parent / "images" / filename
                    
                    if not image_path.exists():
                        logger.warning(f"⚠️ 图片文件不存在: {image_path}")
                        continue
                    
                    # 获取语义上下文
                    from hybrid_diagram_reasoner import get_semantic_context_for_figure
                    context = get_semantic_context_for_figure(content, figure['caption'])
                    
                    # 分析图片
                    diagram_analysis = self.reasoner.analyze_diagram_with_gpt4o(
                        str(image_path), figure['caption'], context
                    )
                    
                    # 生成训练数据
                    from hybrid_diagram_reasoner import generate_diagram_description_with_o3, generate_thinking_with_o3
                    summary_result = generate_diagram_description_with_o3(
                        figure['caption'], context
                    )
                    
                    thinking_result = generate_thinking_with_o3(
                        figure['caption'], context, json.dumps(diagram_analysis, ensure_ascii=False)
                    )
                    
                    # 构建训练数据
                    training_item = {
                        "data_quality": "valid",
                        "quality_issues": [],
                        "stage1_input": {
                            "context": context,
                            "caption": figure['caption']
                        },
                        "stage2_input": {
                            "diagram_description_long": summary_result.get('diagram_description_long', ''),
                            "diagram_description_short": summary_result.get('diagram_description_short', '')
                        },
                        "stage2_output": {
                            "thinking_long": thinking_result.get('thinking_long', ''),
                            "thinking_short": thinking_result.get('thinking_short', ''),
                            "image_path": str(image_path)
                        }
                    }
                    
                    training_data.append(training_item)
                    
                    # 构建judge数据
                    judge_item = {
                        "image_path": str(image_path),
                        "caption": figure['caption'],
                        "context": context,
                        "diagram_analysis": diagram_analysis,
                        "diagram_description_long": summary_result.get('diagram_description_long', ''),
                        "diagram_description_short": summary_result.get('diagram_description_short', ''),
                        "thinking_long": thinking_result.get('thinking_long', ''),
                        "thinking_short": thinking_result.get('thinking_short', '')
                    }
                    
                    judge_data.append(judge_item)
                    
                except Exception as e:
                    logger.error(f"❌ 处理图片时出错 {figure.get('src', 'unknown')}: {e}")
                    continue
            
            # 保存单个文件的结果
            self.save_file_results(training_data, judge_data, output_dir, markdown_file.stem)
            
            return {
                "training_data": training_data,
                "judge_data": judge_data,
                "statistics": {
                    "total_figures": len(figures),
                    "diagram_figures": len(diagram_figures),
                    "processed_figures": len(training_data)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 处理文件时出错: {e}")
            return None
    
    def save_file_results(self, training_data: List[Dict], judge_data: List[Dict], 
                         output_dir: Path, file_name: str):
        """保存单个文件的结果"""
        # 保存训练数据
        training_file = output_dir / f"{file_name}_training_data.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        # 保存judge数据
        judge_file = output_dir / f"{file_name}_judge_data.json"
        with open(judge_file, 'w', encoding='utf-8') as f:
            json.dump(judge_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 保存结果到: {output_dir}")
    
    def process_parallel(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """并行处理所有markdown文件"""
        start_time = time.time()
        
        # 初始化reasoner
        if not self.initialize_reasoner():
            return {"error": "初始化reasoner失败"}
        
        # 查找所有markdown文件
        markdown_files = self.find_markdown_files(input_dir)
        if not markdown_files:
            return {"error": "没有找到markdown文件"}
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 开始并行处理 {len(markdown_files)} 个文件，使用 {self.max_workers} 个线程")
        
        # 并行处理
        results = []
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.process_single_file, file, output_path): file 
                for file in markdown_files
            }
            
            # 收集结果
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["status"] == "failed":
                        failed_files.append(result)
                        logger.error(f"❌ 文件处理失败: {file.name}")
                    else:
                        logger.info(f"✅ 文件处理完成: {file.name}")
                        
                except Exception as e:
                    logger.error(f"❌ 处理文件时出现异常 {file.name}: {e}")
                    failed_files.append({
                        "file_path": str(file),
                        "file_name": file.name,
                        "status": "failed",
                        "error": str(e)
                    })
        
        # 合并所有结果
        all_training_data = []
        all_judge_data = []
        
        for result in results:
            if result["status"] == "completed":
                all_training_data.extend(result.get("training_data", []))
                all_judge_data.extend(result.get("judge_data", []))
        
        # 保存合并结果
        self.save_combined_results(all_training_data, all_judge_data, results, output_path)
        
        # 生成统计信息
        total_time = time.time() - start_time
        statistics = {
            "total_files": len(markdown_files),
            "successful_files": len(results) - len(failed_files),
            "failed_files": len(failed_files),
            "total_training_items": len(all_training_data),
            "total_judge_items": len(all_judge_data),
            "total_processing_time": total_time,
            "average_time_per_file": total_time / len(markdown_files) if markdown_files else 0
        }
        
        logger.info(f"🎉 并行处理完成！")
        logger.info(f"📊 统计信息: {statistics}")
        
        return {
            "statistics": statistics,
            "results": results,
            "failed_files": failed_files,
            "all_training_data": all_training_data,
            "all_judge_data": all_judge_data
        }
    
    def save_combined_results(self, training_data: List[Dict], judge_data: List[Dict], 
                            results: List[Dict], output_dir: Path):
        """保存合并的结果"""
        # 保存合并的训练数据
        training_file = output_dir / "combined_training_data.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        # 保存合并的judge数据
        judge_file = output_dir / "combined_judge_data.json"
        with open(judge_file, 'w', encoding='utf-8') as f:
            json.dump(judge_data, f, ensure_ascii=False, indent=2)
        
        # 保存处理报告
        report_file = output_dir / "processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 合并结果已保存到: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="并行处理markdown文件中的图表")
    parser.add_argument("--input_dir", "-i", required=True, help="输入markdown文件目录")
    parser.add_argument("--output_dir", "-o", required=True, help="输出目录")
    parser.add_argument("--workers", "-w", type=int, default=4, help="并行工作线程数")
    parser.add_argument("--api_source", "-a", choices=["papyrus", "azure"], default="papyrus", 
                       help="API源选择")
    
    args = parser.parse_args()
    
    # 选择API源
    api_source = APISource.PAPYRUS if args.api_source == "papyrus" else APISource.AZURE_OPENAI
    
    # 创建处理器
    processor = ParallelDiagramProcessor(api_source=api_source, max_workers=args.workers)
    
    # 开始处理
    result = processor.process_parallel(args.input_dir, args.output_dir)
    
    if "error" in result:
        logger.error(f"❌ 处理失败: {result['error']}")
        sys.exit(1)
    else:
        logger.info("🎉 所有处理完成！")


if __name__ == "__main__":
    main()
