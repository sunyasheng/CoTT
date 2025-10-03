#!/usr/bin/env python3
"""
简单的图表分析运行脚本
使用示例：python run_chart_analysis.py path/to/chart.png
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("使用方法: python run_chart_analysis.py <图表图片路径>")
        print("示例: python run_chart_analysis.py ../CoTT/debug/papers_latex/arXiv-2509.11171v1/samples/gs.png")
        return
    
    image_path = sys.argv[1]
    
    # 检查图片是否存在
    if not Path(image_path).exists():
        print(f"❌ 图片不存在: {image_path}")
        return
    
    # 设置输出目录
    image_name = Path(image_path).stem
    output_dir = f"./chart_analysis_{image_name}"
    
    # 检查环境变量
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("❌ 请设置环境变量 AZURE_OPENAI_API_KEY")
        print("   export AZURE_OPENAI_API_KEY='your-api-key'")
        return
    
    # 构建命令
    cmd = [
        "/Users/suny0a/anaconda3/envs/scientist.sh/bin/python",  # 使用指定的Python环境
        "universal_chart_processor.py",
        "--image", image_path,
        "--output_dir", output_dir
    ]
    
    print(f"🔍 分析图表: {image_path}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🚀 运行命令: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # 运行分析
        result = subprocess.run(cmd, cwd=Path(__file__).parent, text=True)
        
        if result.returncode == 0:
            print("\n✅ 分析完成！")
            print(f"📂 查看结果: {output_dir}")
            
            # 显示生成的文件
            output_path = Path(output_dir)
            if output_path.exists():
                files = list(output_path.glob("*"))
                print(f"\n📋 生成的文件 ({len(files)}个):")
                for file_path in sorted(files):
                    size = file_path.stat().st_size
                    print(f"   • {file_path.name} ({size} bytes)")
                
                # 检查是否有matplotlib代码
                py_files = list(output_path.glob("*_matplotlib.py"))
                if py_files:
                    print(f"\n🐍 运行重现代码:")
                    print(f"   cd {output_dir}")
                    print(f"   python {py_files[0].name}")
        else:
            print(f"\n❌ 分析失败，退出代码: {result.returncode}")
            
    except Exception as e:
        print(f"❌ 运行出错: {e}")

if __name__ == "__main__":
    main()
