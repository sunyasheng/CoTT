#!/bin/bash
# 服务器批量处理脚本
# 用于处理 /blob/yasheng/Paper2Fig100k_flux_train 目录下的所有图片

# 设置路径
INPUT_DIR="/blob/yasheng/Paper2Fig100k_flux_train"
OUTPUT_DIR="/blob/yasheng/Paper2Fig100k_flux_train_dpg"  # 修改为你想要的输出目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="python"  # 或者使用完整路径，如 /path/to/python

# API 配置（从环境变量读取）
API_TYPE="${API_TYPE:-azure}"  # 默认使用 azure

# 处理参数
MAX_IMAGES="${MAX_IMAGES:-}"  # 空表示处理所有，可以设置如 1000 来测试
START_IDX="${START_IDX:-0}"   # 从哪个索引开始（用于断点续传）
BATCH_SIZE="${BATCH_SIZE:-100}"  # 每处理多少张保存一次checkpoint

# 运行命令
cd "$SCRIPT_DIR" || exit 1

echo "=========================================="
echo "Batch DPG Processing"
echo "=========================================="
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "API Type: $API_TYPE"
echo "Max Images: ${MAX_IMAGES:-All}"
echo "Start Index: $START_IDX"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="

$PYTHON_CMD batch_process_directory.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --api-type "$API_TYPE" \
    ${MAX_IMAGES:+--max-images $MAX_IMAGES} \
    --start-idx "$START_IDX" \
    --batch-size "$BATCH_SIZE" \
    --summary-output "$OUTPUT_DIR/batch_dpg_summary.json"

echo ""
echo "=========================================="
echo "Processing completed!"
echo "Summary saved to: $OUTPUT_DIR/batch_dpg_summary.json"
echo "=========================================="

