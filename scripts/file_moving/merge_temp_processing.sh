#!/usr/bin/env bash
# merge_temp_processing.sh
# 合并 arxiv_dataset 中的 temp_processing_shard_* 到统一的 temp_processing 目录
# Author: suny0a

set -euo pipefail

########################################
# 参数解析
########################################
DATASET_ROOT=${1:-""}

if [[ -z "$DATASET_ROOT" ]]; then
  echo "用法: $0 <DATASET_ROOT>"
  echo "示例: $0 /home/suny0a/ibex_root/arxiv_dataset"
  exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "目录不存在: $DATASET_ROOT"
  exit 1
fi

cd "$DATASET_ROOT"

########################################
# 创建目标目录
########################################
mkdir -p temp_processing/md temp_processing/pdf
echo "[✓] 目标目录 temp_processing/{md,pdf} 已创建/存在"

########################################
# 合并各 shard
########################################
shards=(temp_processing_shard_*)
if [[ ${#shards[@]} -eq 1 && "${shards[0]}" == "temp_processing_shard_*" ]]; then
  echo "未找到 temp_processing_shard_* 目录"
  exit 1
fi

for shard in "${shards[@]}"; do
  echo "⏳ 合并 $shard ..."
  # 合并 md
  rsync -a --ignore-existing "$shard/md/"  temp_processing/md/
  # 合并 pdf
  rsync -a --ignore-existing "$shard/pdf/" temp_processing/pdf/
done
echo "[✓] 文件合并完成"

########################################
# 合并 pdf 列表文本
########################################
echo "⏳ 合并 shard_*_pdfs.txt ..."
cat temp_processing_shard_*/shard_*_pdfs.txt > temp_processing/all_pdfs.txt
echo "[✓] 列表文件 temp_processing/all_pdfs.txt 已生成"

########################################
# 统计数量
########################################
MD_COUNT=$(find temp_processing/md  -type f | wc -l)
PDF_COUNT=$(find temp_processing/pdf -type f | wc -l)
echo "文件统计:"
echo "  md  : $MD_COUNT"
echo "  pdf : $PDF_COUNT"

########################################
# 询问是否删除原分片
########################################
read -p "是否删除原始 temp_processing_shard_* 目录？(y/N): " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
  echo "⏳ 正在删除 ..."
  rm -rf temp_processing_shard_*
  echo "[✓] 已删除原始分片目录"
else
  echo "已保留原始分片目录"
fi

echo "全部完成 ✅"
