#!/bin/bash

# 并行图表处理器运行脚本

echo "🚀 启动并行图表处理器"
echo "=================================="

# 设置默认参数
INPUT_DIR="${1:-/dev/shm/yasheng/md}"
OUTPUT_DIR="${2:-./parallel_output}"
WORKERS="${3:-4}"
API_SOURCE="${4:-papyrus}"

echo "📁 输入目录: $INPUT_DIR"
echo "📂 输出目录: $OUTPUT_DIR"
echo "👥 工作线程数: $WORKERS"
echo "🔗 API源: $API_SOURCE"
echo ""

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行并行处理器
echo "🔄 开始处理..."
python parallel_diagram_processor.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    --api_source "$API_SOURCE"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 处理完成！"
    echo "📊 查看结果:"
    echo "   - 合并训练数据: $OUTPUT_DIR/combined_training_data.json"
    echo "   - 合并judge数据: $OUTPUT_DIR/combined_judge_data.json"
    echo "   - 处理报告: $OUTPUT_DIR/processing_report.json"
    echo "   - 日志文件: parallel_processor.log"
    
    # 显示统计信息
    if [ -f "$OUTPUT_DIR/processing_report.json" ]; then
        echo ""
        echo "📈 处理统计:"
        python -c "
import json
with open('$OUTPUT_DIR/processing_report.json', 'r') as f:
    results = json.load(f)
    
successful = len([r for r in results if r['status'] == 'completed'])
failed = len([r for r in results if r['status'] == 'failed'])
total_training = sum(len(r.get('training_data', [])) for r in results if r['status'] == 'completed')
total_judge = sum(len(r.get('judge_data', [])) for r in results if r['status'] == 'completed')

print(f'   成功处理: {successful} 个文件')
print(f'   失败文件: {failed} 个文件')
print(f'   训练数据: {total_training} 条')
print(f'   Judge数据: {total_judge} 条')
"
    fi
else
    echo "❌ 处理失败，请查看日志文件: parallel_processor.log"
    exit 1
fi
