#!/usr/bin/env python3
"""
Universal Chart Processor - 通用图表处理工具

功能：
1. 分析各种类型的图表（折线图、柱状图、散点图、饼图等）
2. 分离内容（原始数据）和格式（样式、布局）
3. 保存原始数据为JSON/CSV格式
4. 生成matplotlib代码重现图表格式
5. 支持复杂图表：双轴、子图、混合图表等

使用方法：
    python universal_chart_processor.py --image chart.png --output_dir ./results
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import csv
import requests
from PIL import Image


# Universal Chart Analysis Prompt
UNIVERSAL_CHART_PROMPT = """
Extract ALL data from this chart. Identify chart type and extract complete data accurately.

Return JSON format:

{
  "chart_info": {
    "type": "line/bar/scatter/pie/histogram/heatmap/box/violin/area/bubble/radar/treemap/sankey/gantt/waterfall/funnel/mixed",
    "title": "chart title",
    "description": "what the chart shows"
  },
  "data_series": [
    {
      "series_id": "series_1",
      "name": "series name", 
      "type": "line/bar/scatter/pie/area/heatmap/box/other",
      "data": {
        "x": [x_values],
        "y": [y_values],
        "z": [z_values_for_3d_or_heatmap],
        "categories": [category_names],
        "values": [numerical_values],
        "labels": [text_labels],
        "sizes": [bubble_sizes],
        "colors": [color_values_for_heatmap],
        "matrix": [[row1], [row2], [row3]],
        "coordinates": [[x1,y1], [x2,y2]]
      },
      "axis": "primary/secondary/none"
    }
  ],
  "axes": [
    {
      "x_axis": {"label": "x_label", "range": [min, max], "type": "linear/categorical/datetime"},
      "y_axis": {"label": "y_label", "range": [min, max], "type": "linear/log"},
      "y_axis_secondary": {"label": "right_y_label", "range": [min, max]}
    }
  ],
  "legend": {"entries": ["item1", "item2"]},
  "annotations": [{"text": "annotation", "position": [x, y]}]
}

**REQUIREMENTS:**
1. **EXTRACT ALL DATA** - Every visible data point, bar, pie slice, heatmap cell, etc.
2. **CHART TYPE SPECIFIC**:
   - Line/Scatter: x,y coordinates for each point
   - Bar: categories and values
   - **STACKED BAR**: For each category, measure individual segment heights carefully
   - Pie: labels and values (percentages or counts)
   - Heatmap: matrix data or x,y,z coordinates with color values
   - Box plot: quartiles, outliers, whiskers
   - Histogram: bins and frequencies
3. **STACKED CHARTS SPECIAL ATTENTION**:
   - Identify each colored segment in the stack
   - Measure the HEIGHT of each segment (not cumulative position)
   - Read values from the y-axis scale for each segment boundary
   - Calculate individual segment values by subtracting positions
4. **DUAL AXIS**: Mark series as "primary"/"secondary" if multiple y-axes
5. **PRECISION**: Extract exact numerical values as shown
6. **COMPLETENESS**: Don't miss any data series or elements

Return ONLY valid JSON.
"""


def encode_image_b64(image_path: Path, max_size: int = 1024) -> str:
    """将图片编码为base64格式"""
    img = Image.open(image_path)
    
    # 如果图片太大就缩放
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # 转换为RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 保存为字节流
    import io
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    data = buffer.getvalue()
    
    return base64.b64encode(data).decode("utf-8")


def analyze_chart_with_gpt(image_path: Path, azure_config: Dict) -> Dict:
    """使用GPT分析图表"""
    
    # 编码图片
    image_b64 = encode_image_b64(image_path)
    
    # 构建API端点
    url = (
        f"{azure_config['endpoint']}openai/deployments/{azure_config['deployment']}/chat/completions?"
        f"api-version={azure_config['api_version']}"
    )
    
    headers = {
        "Content-Type": "application/json", 
        "api-key": azure_config["api_key"]
    }
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": UNIVERSAL_CHART_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        
        # 解析JSON响应
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # 尝试从响应中提取JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": f"Failed to parse JSON: {e}", "raw_response": content}
                
    except Exception as e:
        return {"error": f"API call failed: {e}"}


def save_raw_data(chart_analysis: Dict, output_dir: Path, image_name: str) -> Dict[str, str]:
    """保存原始数据（内容部分）"""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(image_name).stem
    saved_files = {}
    
    # 1. 保存完整分析结果
    analysis_path = output_dir / f"{base_name}_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(chart_analysis, f, indent=2, ensure_ascii=False)
    saved_files['complete_analysis'] = str(analysis_path)
    
    # 2. 分离并保存纯数据
    raw_data = {
        "chart_info": chart_analysis.get("chart_info", {}),
        "data_series": []
    }
    
    # 提取每个数据系列的原始数据
    for series in chart_analysis.get("data_series", []):
        series_data = {
            "series_id": series.get("series_id"),
            "name": series.get("name"),
            "type": series.get("type"),
            "data": series.get("data", {}),
            "subplot_index": series.get("subplot_index", 0)
        }
        raw_data["data_series"].append(series_data)
    
    # 保存纯数据
    raw_data_path = output_dir / f"{base_name}_raw_data.json"
    with open(raw_data_path, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)
    saved_files['raw_data'] = str(raw_data_path)
    
    # 3. 保存每个数据系列为CSV
    for i, series in enumerate(chart_analysis.get("data_series", [])):
        series_name = series.get("name", f"series_{i}")
        safe_name = re.sub(r'[^\w\-_.]', '_', series_name)
        
        data = series.get("data", {})
        csv_path = output_dir / f"{base_name}_{safe_name}.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 根据数据类型写入不同格式
            if "x" in data and "y" in data:
                # 散点图/折线图数据
                writer.writerow(["x", "y"])
                x_data = data["x"]
                y_data = data["y"]
                for j in range(min(len(x_data), len(y_data))):
                    writer.writerow([x_data[j], y_data[j]])
            elif "categories" in data and "values" in data:
                # 柱状图数据
                writer.writerow(["category", "value"])
                cats = data["categories"]
                vals = data["values"]
                for j in range(min(len(cats), len(vals))):
                    writer.writerow([cats[j], vals[j]])
            elif "labels" in data and "values" in data:
                # 饼图数据
                writer.writerow(["label", "value"])
                labels = data["labels"]
                values = data["values"]
                for j in range(min(len(labels), len(values))):
                    writer.writerow([labels[j], values[j]])
        
        saved_files[f'series_{i}_csv'] = str(csv_path)
    
    return saved_files


def save_style_info(chart_analysis: Dict, output_dir: Path, image_name: str) -> str:
    """保存样式信息（格式部分）"""
    base_name = Path(image_name).stem
    
    # 提取样式相关信息
    style_info = {
        "layout": chart_analysis.get("layout", {}),
        "axes": chart_analysis.get("axes", []),
        "series_styles": [],
        "annotations": chart_analysis.get("annotations", []),
        "legend": chart_analysis.get("legend", {}),
        "global_style": chart_analysis.get("style", {})
    }
    
    # 提取每个数据系列的样式
    for series in chart_analysis.get("data_series", []):
        series_style = {
            "series_id": series.get("series_id"),
            "name": series.get("name"),
            "type": series.get("type"),
            "style": series.get("style", {}),
            "axis": series.get("axis", "primary"),
            "subplot_index": series.get("subplot_index", 0)
        }
        style_info["series_styles"].append(series_style)
    
    # 保存样式信息
    style_path = output_dir / f"{base_name}_style.json"
    with open(style_path, 'w', encoding='utf-8') as f:
        json.dump(style_info, f, indent=2, ensure_ascii=False)
    
    return str(style_path)


def generate_matplotlib_code(chart_analysis: Dict, image_name: str) -> str:
    """生成matplotlib代码重现图表"""
    
    base_name = Path(image_name).stem
    code_lines = [
        "#!/usr/bin/env python3",
        "# -*- coding: utf-8 -*-",
        "\"\"\"",
        f"自动生成的matplotlib代码 - 重现图表: {image_name}",
        "\"\"\"",
        "",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import pandas as pd",
        "from matplotlib import colors",
        "import seaborn as sns",
        "",
        "# 设置中文字体支持",
        "plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']",
        "plt.rcParams['axes.unicode_minus'] = False",
        "",
    ]
    
    # 获取布局信息
    layout = chart_analysis.get("layout", {})
    subplots = layout.get("subplots", {"rows": 1, "cols": 1})
    
    # 创建图形和子图
    fig_size = layout.get("figure_size", [10, 6])
    code_lines.extend([
        f"# 创建图形和子图",
        f"fig, axes = plt.subplots({subplots['rows']}, {subplots['cols']}, figsize=({fig_size[0]}, {fig_size[1]}))",
        ""
    ])
    
    # 如果只有一个子图，确保axes是数组
    if subplots['rows'] * subplots['cols'] == 1:
        code_lines.append("if not isinstance(axes, np.ndarray):")
        code_lines.append("    axes = np.array([axes])")
        code_lines.append("")
    
    # 设置全局样式
    global_style = chart_analysis.get("style", {})
    if global_style.get("theme"):
        code_lines.append(f"plt.style.use('{global_style['theme']}')")
    
    # 颜色调色板
    color_palette = global_style.get("color_palette", [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])
    code_lines.extend([
        f"# 颜色调色板",
        f"colors_palette = {color_palette}",
        ""
    ])
    
    # 按子图分组处理数据系列
    subplot_series = {}
    for series in chart_analysis.get("data_series", []):
        subplot_idx = series.get("subplot_index", 0)
        if subplot_idx not in subplot_series:
            subplot_series[subplot_idx] = []
        subplot_series[subplot_idx].append(series)
    
    # 为每个子图生成代码
    for subplot_idx, series_list in subplot_series.items():
        code_lines.extend([
            f"# === 子图 {subplot_idx} ===",
            f"ax = axes[{subplot_idx}] if len(axes) > 1 else axes[0]",
            ""
        ])
        
        # 检查是否需要双Y轴
        has_secondary = any(s.get("axis") == "secondary" for s in series_list)
        if has_secondary:
            code_lines.extend([
                "# 创建双Y轴",
                "ax2 = ax.twinx()",
                ""
            ])
        
        # 绘制每个数据系列
        for i, series in enumerate(series_list):
            series_name = series.get("name", f"Series {i+1}")
            series_type = series.get("type", "line")
            data = series.get("data", {})
            style = series.get("style", {})
            axis_type = series.get("axis", "primary")
            
            # 选择轴
            ax_var = "ax2" if axis_type == "secondary" else "ax"
            
            # 获取样式参数
            color = style.get("color", color_palette[i % len(color_palette)])
            line_style = style.get("line_style", "-")
            line_width = style.get("line_width", 2)
            marker = style.get("marker", "o")
            marker_size = style.get("marker_size", 6)
            alpha = style.get("alpha", 1.0)
            
            code_lines.extend([
                f"# 数据系列: {series_name}",
            ])
            
            # 根据图表类型生成不同的绘图代码
            if series_type == "line":
                if "x" in data and "y" in data:
                    code_lines.extend([
                        f"x_{i} = {data['x']}",
                        f"y_{i} = {data['y']}",
                        f"{ax_var}.plot(x_{i}, y_{i}, label='{series_name}', color='{color}', ",
                        f"         linestyle='{line_style}', linewidth={line_width}, ",
                        f"         marker='{marker}', markersize={marker_size}, alpha={alpha})"
                    ])
            
            elif series_type == "bar":
                if "categories" in data and "values" in data:
                    code_lines.extend([
                        f"categories_{i} = {data['categories']}",
                        f"values_{i} = {data['values']}",
                        f"x_pos_{i} = np.arange(len(categories_{i}))",
                        f"{ax_var}.bar(x_pos_{i}, values_{i}, label='{series_name}', ",
                        f"       color='{color}', alpha={alpha})",
                        f"{ax_var}.set_xticks(x_pos_{i})",
                        f"{ax_var}.set_xticklabels(categories_{i})"
                    ])
            
            elif series_type == "scatter":
                if "x" in data and "y" in data:
                    code_lines.extend([
                        f"x_{i} = {data['x']}",
                        f"y_{i} = {data['y']}",
                        f"{ax_var}.scatter(x_{i}, y_{i}, label='{series_name}', color='{color}', ",
                        f"          s={marker_size**2}, alpha={alpha}, marker='{marker}')"
                    ])
            
            elif series_type == "pie":
                if "labels" in data and "values" in data:
                    code_lines.extend([
                        f"labels_{i} = {data['labels']}",
                        f"values_{i} = {data['values']}",
                        f"{ax_var}.pie(values_{i}, labels=labels_{i}, autopct='%1.1f%%', ",
                        f"       colors=[colors_palette[j % len(colors_palette)] for j in range(len(values_{i}))])"
                    ])
            
            elif series_type == "area":
                if "x" in data and "y" in data:
                    fill_alpha = style.get("fill_alpha", 0.3)
                    code_lines.extend([
                        f"x_{i} = {data['x']}",
                        f"y_{i} = {data['y']}",
                        f"{ax_var}.fill_between(x_{i}, y_{i}, label='{series_name}', ",
                        f"                color='{color}', alpha={fill_alpha})",
                        f"{ax_var}.plot(x_{i}, y_{i}, color='{color}', linewidth={line_width})"
                    ])
            
            code_lines.append("")
        
        # 设置轴标签和标题
        axes_info = None
        for axis_info in chart_analysis.get("axes", []):
            if axis_info.get("subplot_index", 0) == subplot_idx:
                axes_info = axis_info
                break
        
        if axes_info:
            x_axis = axes_info.get("x_axis", {})
            y_axis = axes_info.get("y_axis", {})
            y_axis_sec = axes_info.get("y_axis_secondary", {})
            
            if x_axis.get("label"):
                code_lines.append(f"ax.set_xlabel('{x_axis['label']}')")
            if y_axis.get("label"):
                code_lines.append(f"ax.set_ylabel('{y_axis['label']}')")
            if has_secondary and y_axis_sec.get("label"):
                code_lines.append(f"ax2.set_ylabel('{y_axis_sec['label']}')")
            
            # 设置轴范围
            if x_axis.get("range"):
                x_range = x_axis["range"]
                code_lines.append(f"ax.set_xlim({x_range[0]}, {x_range[1]})")
            if y_axis.get("range"):
                y_range = y_axis["range"]
                code_lines.append(f"ax.set_ylim({y_range[0]}, {y_range[1]})")
            if has_secondary and y_axis_sec.get("range"):
                y2_range = y_axis_sec["range"]
                code_lines.append(f"ax2.set_ylim({y2_range[0]}, {y2_range[1]})")
            
            # 网格
            if x_axis.get("grid") or y_axis.get("grid"):
                grid_style = chart_analysis.get("style", {}).get("grid_style", {})
                grid_alpha = grid_style.get("alpha", 0.3)
                code_lines.append(f"ax.grid(True, alpha={grid_alpha})")
        
        code_lines.append("")
    
    # 设置图表标题
    chart_info = chart_analysis.get("chart_info", {})
    if chart_info.get("title"):
        code_lines.append(f"fig.suptitle('{chart_info['title']}', fontsize=16, fontweight='bold')")
    
    # 设置子图标题
    subplot_titles = layout.get("subplots", {}).get("subplot_titles", [])
    for i, title in enumerate(subplot_titles):
        if title:
            code_lines.append(f"axes[{i}].set_title('{title}')")
    
    # 图例
    legend_info = chart_analysis.get("legend", {})
    if legend_info.get("show", True) and len(chart_analysis.get("data_series", [])) > 1:
        legend_pos = legend_info.get("position", "best")
        if has_secondary:
            code_lines.extend([
                "# 合并双轴图例",
                "lines1, labels1 = ax.get_legend_handles_labels()",
                "lines2, labels2 = ax2.get_legend_handles_labels()",
                f"ax.legend(lines1 + lines2, labels1 + labels2, loc='{legend_pos}')"
            ])
        else:
            code_lines.append(f"plt.legend(loc='{legend_pos}')")
    
    # 添加注释
    for annotation in chart_analysis.get("annotations", []):
        ann_text = annotation.get("text", "")
        ann_pos = annotation.get("position", [0, 0])
        ann_style = annotation.get("style", {})
        fontsize = ann_style.get("fontsize", 12)
        color = ann_style.get("color", "black")
        
        code_lines.append(f"ax.annotate('{ann_text}', xy=({ann_pos[0]}, {ann_pos[1]}), ")
        code_lines.append(f"            fontsize={fontsize}, color='{color}')")
    
    # 最终调整和保存
    code_lines.extend([
        "",
        "# 调整布局",
        "plt.tight_layout()",
        "",
        "# 保存和显示",
        f"plt.savefig('{base_name}_recreated.png', dpi=300, bbox_inches='tight')",
        f"plt.savefig('{base_name}_recreated.pdf', bbox_inches='tight')",
        "plt.show()",
        "",
        "print('图表已重新生成并保存！')"
    ])
    
    return "\n".join(code_lines)


def main():
    parser = argparse.ArgumentParser(description="通用图表处理工具 - 分离内容和格式")
    parser.add_argument("--image", required=True, help="图表图片路径")
    parser.add_argument("--output_dir", default="./chart_results", help="输出目录")
    
    # Azure配置
    parser.add_argument("--azure_endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT", 
                       "https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com/"))
    parser.add_argument("--azure_api_key", default=os.getenv("AZURE_OPENAI_API_KEY", ""))
    parser.add_argument("--deployment", default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"))
    parser.add_argument("--api_version", default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"))
    
    args = parser.parse_args()
    
    # 验证输入
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"图片未找到: {image_path}")
    
    output_dir = Path(args.output_dir)
    
    # Azure配置
    azure_config = {
        "endpoint": args.azure_endpoint,
        "api_key": args.azure_api_key,
        "deployment": args.deployment,
        "api_version": args.api_version
    }
    
    print(f"🔍 分析图表: {image_path}")
    print(f"📁 输出目录: {output_dir}")
    
    # 步骤1: 分析图表
    print("\n1️⃣ 使用GPT分析图表...")
    chart_analysis = analyze_chart_with_gpt(image_path, azure_config)
    
    if chart_analysis.get('error'):
        print(f"❌ 错误: {chart_analysis['error']}")
        return
    
    chart_info = chart_analysis.get('chart_info', {})
    print(f"   📊 图表类型: {chart_info.get('type', '未知')}")
    print(f"   📝 标题: {chart_info.get('title', '无标题')}")
    print(f"   📈 数据系列: {len(chart_analysis.get('data_series', []))}个")
    
    # 步骤2: 保存原始数据（内容）
    print("\n2️⃣ 保存原始数据（内容部分）...")
    data_files = save_raw_data(chart_analysis, output_dir, image_path.name)
    for file_type, file_path in data_files.items():
        print(f"   💾 {file_type}: {Path(file_path).name}")
    
    # 步骤3: 保存样式信息（格式）
    print("\n3️⃣ 保存样式信息（格式部分）...")
    style_file = save_style_info(chart_analysis, output_dir, image_path.name)
    print(f"   🎨 样式文件: {Path(style_file).name}")
    
    # 步骤4: 生成matplotlib代码
    print("\n4️⃣ 生成matplotlib重现代码...")
    matplotlib_code = generate_matplotlib_code(chart_analysis, image_path.name)
    code_path = output_dir / f"{image_path.stem}_matplotlib.py"
    with open(code_path, 'w', encoding='utf-8') as f:
        f.write(matplotlib_code)
    print(f"   🐍 Python代码: {code_path.name}")
    
    print(f"\n✅ 处理完成！")
    print(f"📂 所有文件保存在: {output_dir}")
    print(f"\n📋 文件说明:")
    print(f"   • *_raw_data.json    - 纯数据内容")
    print(f"   • *_style.json       - 样式格式信息") 
    print(f"   • *_matplotlib.py    - 重现代码")
    print(f"   • *.csv              - 各数据系列CSV文件")
    
    # 运行提示
    print(f"\n🚀 运行重现代码:")
    print(f"   cd {output_dir}")
    print(f"   python {code_path.name}")


if __name__ == "__main__":
    main()
