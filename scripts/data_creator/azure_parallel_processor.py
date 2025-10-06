#!/usr/bin/env python3
"""
Azure简单并行处理器 - 直接复用azure_diagram_reasoner.py的逻辑

功能：
1. 扫描指定目录下的所有markdown文件
2. 使用多线程并行处理每个文件
3. 直接调用azure_diagram_reasoner.py中的test_smart_markdown_paper函数
4. 合并所有结果

使用方法：
python azure_parallel_processor.py --input_dir /path/to/markdown/files --output_dir /path/to/output --workers 4
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

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from azure_diagram_reasoner import test_smart_markdown_paper
except ImportError as e:
    print(f"❌ 无法导入azure_diagram_reasoner: {e}")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('azure_simple_parallel_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AzureSimpleParallelProcessor:
    """Azure简单并行处理器 - 直接复用azure_diagram_reasoner逻辑"""
    
    def __init__(self, max_workers: int = 4, skip_existing: bool = True):
        self.max_workers = max_workers
        self.skip_existing = skip_existing
        
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
    
    def check_existing_data(self, output_dir: Path, file_name: str) -> bool:
        """检查是否已存在训练数据文件"""
        file_output_dir = output_dir / file_name
        training_file = file_output_dir / f"{file_name}_training_data.json"
        judge_file = file_output_dir / f"{file_name}_judge_data.json"
        
        # 检查两个文件是否都存在且不为空
        if training_file.exists() and judge_file.exists():
            try:
                # 检查文件内容是否有效
                with open(training_file, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
                with open(judge_file, 'r', encoding='utf-8') as f:
                    judge_data = json.load(f)
                
                # 检查数据是否为空
                if training_data and judge_data:
                    logger.info(f"✅ 发现已存在的训练数据: {file_name}")
                    return True
                else:
                    logger.info(f"⚠️ 训练数据文件存在但为空: {file_name}")
                    return False
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"⚠️ 训练数据文件损坏: {file_name}, 错误: {e}")
                return False
        
        return False

    def load_existing_data(self, output_dir: Path, file_name: str) -> Dict[str, Any]:
        """加载已存在的训练数据"""
        file_output_dir = output_dir / file_name
        training_file = file_output_dir / f"{file_name}_training_data.json"
        judge_file = file_output_dir / f"{file_name}_judge_data.json"
        
        try:
            with open(training_file, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            with open(judge_file, 'r', encoding='utf-8') as f:
                judge_data = json.load(f)
            
            return {
                "training_data": training_data,
                "judge_data": judge_data,
                "statistics": {
                    "total_figures": len(training_data),  # 估算
                    "diagram_figures": len(training_data),
                    "processed_results": len(training_data)
                }
            }
        except Exception as e:
            logger.error(f"❌ 加载已存在数据失败: {file_name}, 错误: {e}")
            return None

    def process_single_file(self, markdown_file: Path, output_dir: Path) -> Dict[str, Any]:
        """处理单个markdown文件 - 直接调用azure_diagram_reasoner函数"""
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
            
            # 注意：跳过的文件已经在主流程中处理，这里只处理需要重新生成的文件
            
            # 直接调用azure_diagram_reasoner中的test_smart_markdown_paper函数
            # 这个函数已经包含了完整的处理逻辑
            paper_dir = markdown_file.parent.parent  # 回到论文根目录
            paper_name = markdown_file.stem
            
            logger.info(f"📁 论文目录: {paper_dir}")
            logger.info(f"📄 论文名称: {paper_name}")
            
            result = test_smart_markdown_paper(paper_dir, paper_name)
            
            if result and "results" in result and result["results"]:
                # 合并所有results中的training_data和judge_data
                all_training_data = []
                all_judge_data = []
                
                for res in result["results"]:
                    if "training_data" in res:
                        all_training_data.append(res["training_data"])
                    if "judge_data" in res:
                        all_judge_data.append(res["judge_data"])
                
                file_result["training_data"] = all_training_data
                file_result["judge_data"] = all_judge_data
                file_result["statistics"] = {
                    "total_figures": result.get("total_figures", 0),
                    "diagram_figures": result.get("diagram_figures", 0),
                    "processed_results": len(result.get("results", []))
                }
                file_result["status"] = "completed"
                
                # 保存单个文件的结果
                self.save_file_results(all_training_data, all_judge_data, file_output_dir, markdown_file.stem)
                
                logger.info(f"✅ 完成处理: {markdown_file.name}")
            else:
                file_result["status"] = "failed"
                file_result["error"] = "处理返回空结果或没有results"
                logger.error(f"❌ 处理失败: {markdown_file.name}")
                
        except Exception as e:
            file_result["status"] = "failed"
            file_result["error"] = str(e)
            logger.error(f"❌ 处理文件时出错 {markdown_file.name}: {e}")
        
        file_result["end_time"] = datetime.now().isoformat()
        file_result["processing_time"] = time.time() - start_time
        
        return file_result
    
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
    
    def scan_existing_files(self, markdown_files: List[Path], output_dir: Path) -> tuple:
        """扫描已存在的文件，返回需要处理和跳过的文件列表"""
        if not self.skip_existing:
            return markdown_files, []
        
        files_to_process = []
        files_to_skip = []
        
        logger.info(f"🔍 扫描已存在的训练数据文件...")
        
        for file in markdown_files:
            if self.check_existing_data(output_dir, file.stem):
                files_to_skip.append(file)
            else:
                files_to_process.append(file)
        
        logger.info(f"📊 扫描结果:")
        logger.info(f"   ⏭️ 将跳过 {len(files_to_skip)} 个已存在数据的文件")
        logger.info(f"   🔄 将处理 {len(files_to_process)} 个新文件")
        
        if files_to_skip:
            logger.info(f"   📋 跳过的文件列表:")
            for file in files_to_skip[:10]:  # 只显示前10个
                logger.info(f"      - {file.name}")
            if len(files_to_skip) > 10:
                logger.info(f"      ... 还有 {len(files_to_skip) - 10} 个文件")
        
        return files_to_process, files_to_skip

    def process_parallel(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """并行处理所有markdown文件"""
        start_time = time.time()
        
        # 查找所有markdown文件
        markdown_files = self.find_markdown_files(input_dir)
        if not markdown_files:
            return {"error": "没有找到markdown文件"}
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 扫描已存在的文件
        files_to_process, files_to_skip = self.scan_existing_files(markdown_files, output_path)
        
        logger.info(f"🚀 开始并行处理 {len(files_to_process)} 个文件，使用 {self.max_workers} 个线程")
        
        # 并行处理
        results = []
        failed_files = []
        skipped_files = []
        
        # 先处理跳过的文件，直接加载已存在的数据
        logger.info(f"📥 加载 {len(files_to_skip)} 个已存在文件的数据...")
        for i, file in enumerate(files_to_skip, 1):
            if i % 1000 == 0:  # 每1000个文件显示一次进度
                logger.info(f"   📥 已加载 {i}/{len(files_to_skip)} 个跳过文件")
            
            existing_data = self.load_existing_data(output_path, file.stem)
            if existing_data:
                skipped_result = {
                    "file_path": str(file),
                    "file_name": file.name,
                    "status": "skipped",
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "processing_time": 0.0,
                    "error": None,
                    "training_data": existing_data["training_data"],
                    "judge_data": existing_data["judge_data"],
                    "statistics": existing_data["statistics"]
                }
                skipped_files.append(skipped_result)
                results.append(skipped_result)
        
        if files_to_skip:
            logger.info(f"✅ 完成加载 {len(skipped_files)} 个跳过文件的数据")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 只提交需要处理的任务
            future_to_file = {
                executor.submit(self.process_single_file, file, output_path): file 
                for file in files_to_process
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
                    elif result["status"] == "skipped":
                        skipped_files.append(result)
                        logger.info(f"⏭️ 文件已跳过: {file.name}")
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
            if result["status"] in ["completed", "skipped"]:
                all_training_data.extend(result.get("training_data", []))
                all_judge_data.extend(result.get("judge_data", []))
        
        # 保存合并结果
        self.save_combined_results(all_training_data, all_judge_data, results, output_path)
        
        # 生成统计信息
        total_time = time.time() - start_time
        successful_files = len([r for r in results if r["status"] == "completed"])
        statistics = {
            "total_files": len(markdown_files),
            "files_to_process": len(files_to_process),
            "files_to_skip": len(files_to_skip),
            "successful_files": successful_files,
            "skipped_files": len(skipped_files),
            "failed_files": len(failed_files),
            "total_training_items": len(all_training_data),
            "total_judge_items": len(all_judge_data),
            "total_processing_time": total_time,
            "average_time_per_file": total_time / len(files_to_process) if files_to_process else 0,
            "skip_existing_enabled": self.skip_existing
        }
        
        logger.info(f"🎉 并行处理完成！")
        logger.info(f"📊 统计信息: {statistics}")
        
        return {
            "statistics": statistics,
            "results": results,
            "failed_files": failed_files,
            "skipped_files": skipped_files,
            "all_training_data": all_training_data,
            "all_judge_data": all_judge_data
        }
    
    def save_combined_results(self, training_data: List[Dict], judge_data: List[Dict], 
                            results: List[Dict], output_dir: Path):
        """保存合并的结果"""
        # 保存合并的训练数据
        training_file = output_dir / "azure_combined_training_data.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        # 保存合并的judge数据
        judge_file = output_dir / "azure_combined_judge_data.json"
        with open(judge_file, 'w', encoding='utf-8') as f:
            json.dump(judge_data, f, ensure_ascii=False, indent=2)
        
        # 保存处理报告
        report_file = output_dir / "azure_processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Azure合并结果已保存到: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Azure简单并行处理markdown文件中的图表")
    parser.add_argument("--input_dir", "-i", required=True, help="输入markdown文件目录")
    parser.add_argument("--output_dir", "-o", required=True, help="输出目录")
    parser.add_argument("--workers", "-w", type=int, default=4, help="并行工作线程数")
    parser.add_argument("--no-skip-existing", action="store_true", 
                       help="禁用跳过已存在训练数据的功能，强制重新处理所有文件")
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = AzureSimpleParallelProcessor(max_workers=args.workers, skip_existing=not args.no_skip_existing)
    
    # 开始处理
    result = processor.process_parallel(args.input_dir, args.output_dir)
    
    if "error" in result:
        logger.error(f"❌ 处理失败: {result['error']}")
        sys.exit(1)
    else:
        logger.info("🎉 所有处理完成！")


if __name__ == "__main__":
    main()
