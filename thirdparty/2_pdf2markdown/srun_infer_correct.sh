#!/usr/bin/env bash

# 正确的远程处理逻辑：
# 本地机器(suny0a@10.64.75.69): 存储PDF文件和输出结果
# 远程机器(集群节点): 提供GPU计算资源
# 工作流程: 本地读取 → 远程处理 → 本地保存

set -euo pipefail

# 本地机器配置（存储机器）
LOCAL_HOST="suny0a@10.64.74.69"
LOCAL_SSH_KEY="~/.ssh/id_rsa"
LOCAL_PDF_DIR="/home/suny0a/arxiv_dataset/pdf/"
LOCAL_OUT_DIR="/home/suny0a/arxiv_dataset/md/"

# 远程机器临时目录（计算节点）
# 使用shard ID确保唯一性
REMOTE_TEMP_DIR="/ibex/user/suny0a/arxiv_dataset/temp_processing_shard"

# Sharding config
TOTAL=0  # 0 means process all PDFs
SHARDS=16  # 16张GPU卡，支持16个并发任务

echo "=== 远程计算资源处理本地存储文件 ==="
echo "本地存储机器: ${LOCAL_HOST}"
echo "PDF目录: ${LOCAL_PDF_DIR}"
echo "输出目录: ${LOCAL_OUT_DIR}"
echo "远程临时目录: ${REMOTE_TEMP_DIR}"

# 1. 从本地机器获取PDF文件列表
echo "1. 从本地机器获取PDF文件列表..."
PDF_LIST_FILE="/ibex/user/suny0a/arxiv_dataset/pdf_list.txt"

# 检查PDF列表文件是否已存在
if [[ -f "${PDF_LIST_FILE}" ]]; then
    echo "✓ PDF列表文件已存在，跳过生成步骤"
    echo "  文件大小: $(du -h "${PDF_LIST_FILE}" | cut -f1)"
    echo "  文件行数: $(wc -l < "${PDF_LIST_FILE}")"
else
    echo "  正在生成PDF文件列表，这可能需要几分钟时间..."
    ssh -i "${LOCAL_SSH_KEY}" "${LOCAL_HOST}" "find '${LOCAL_PDF_DIR}' -name '*.pdf' -type f" > "${PDF_LIST_FILE}"
    echo "✓ PDF文件列表生成完成"
fi

# Count total PDFs with deduplication
echo "2. 统计PDF文件数量并去重..."
COUNT_ALL=$(python3 - <<PY
from pathlib import Path
import re

# 读取本地PDF文件列表
with open('${PDF_LIST_FILE}', 'r') as f:
    all_pdfs = [line.strip() for line in f if line.strip()]

# Deduplicate by paper ID (same logic as batch script)
paper_groups = {}
for pdf_path in all_pdfs:
    filename = Path(pdf_path).name
    match = re.match(r'(\d{4}\.\d{4,5})', filename)
    if match:
        paper_id = match.group(1)
        if paper_id not in paper_groups:
            paper_groups[paper_id] = []
        paper_groups[paper_id].append(pdf_path)

deduped_pdfs = []
for paper_id, versions in sorted(paper_groups.items()):
    versions.sort()
    deduped_pdfs.append(versions[0])

# Shuffle for better load balancing across shards
import random
random.seed(42)  # Fixed seed for reproducibility
random.shuffle(deduped_pdfs)

print(len(deduped_pdfs))
PY
)
echo "✓ PDF统计完成，去重后共 ${COUNT_ALL} 个文件"

if [[ "$COUNT_ALL" -eq 0 ]]; then
  echo "No PDF files found under ${LOCAL_PDF_DIR}" >&2
  exit 2
fi

if [[ "$TOTAL" -eq 0 ]] || [[ "$TOTAL" -gt "$COUNT_ALL" ]]; then
  TOTAL="$COUNT_ALL"
fi

PER_SHARD=$(( (TOTAL + SHARDS - 1) / SHARDS ))

# Count existing outputs
echo "3. 检查已处理的文件..."
echo "  使用批量检查方式，这会更快..."

# 先在远程机器上生成所有可能的输出文件列表
echo "  生成输出文件列表..."
ssh -i "${LOCAL_SSH_KEY}" "${LOCAL_HOST}" "find '${LOCAL_OUT_DIR}' -name '*.md' -type f" > "/ibex/user/suny0a/arxiv_dataset/existing_md_files.txt"

EXISTING=$(python3 - <<PY
from pathlib import Path
import re

# 读取本地PDF文件列表
with open('${PDF_LIST_FILE}', 'r') as f:
    all_pdfs = [line.strip() for line in f if line.strip()]

# 读取已存在的markdown文件列表
with open('/ibex/user/suny0a/arxiv_dataset/existing_md_files.txt', 'r') as f:
    existing_md_files = set(line.strip() for line in f if line.strip())

# Deduplicate by paper ID
paper_groups = {}
for pdf_path in all_pdfs:
    filename = Path(pdf_path).name
    match = re.match(r'(\d{4}\.\d{4,5})', filename)
    if match:
        paper_id = match.group(1)
        if paper_id not in paper_groups:
            paper_groups[paper_id] = []
        paper_groups[paper_id].append(pdf_path)

deduped_pdfs = []
for paper_id, versions in sorted(paper_groups.items()):
    versions.sort()
    deduped_pdfs.append(versions[0])

# Shuffle for better load balancing across shards
import random
random.seed(42)  # Fixed seed for reproducibility
random.shuffle(deduped_pdfs)

# Check how many already have markdown output (本地检查，不需要SSH)
existing = 0
for pdf in deduped_pdfs[:${TOTAL}]:
    pdf_stem = Path(pdf).stem
    expected_md_file = f"${LOCAL_OUT_DIR}/{pdf_stem}/vlm/{pdf_stem}.md"
    if expected_md_file in existing_md_files:
        existing += 1
print(existing)
PY
)
echo "✓ 已处理文件检查完成，共 ${EXISTING} 个文件已处理"

echo "Found ${COUNT_ALL} unique papers (deduplicated)"
echo "Found ${EXISTING}/${TOTAL} PDFs already processed in ${LOCAL_OUT_DIR}"
echo "Launching ${SHARDS} shards over all ${TOTAL} unique papers (≈${PER_SHARD}/shard) using 16 GPUs"

# 创建处理脚本
echo "4. 创建处理脚本..."
PROCESS_SCRIPT="/ibex/user/suny0a/arxiv_dataset/process_shard_correct.sh"
cat > "${PROCESS_SCRIPT}" << 'EOF'
#!/bin/bash
set -euo pipefail

SHARD_ID=$1
START=$2
END=$3
LOCAL_HOST="$4"
LOCAL_PDF_DIR="$5"
LOCAL_OUT_DIR="$6"
REMOTE_TEMP_DIR="$7"
LOCAL_SSH_KEY="~/.ssh/id_rsa"
PDF_LIST_FILE="/ibex/user/suny0a/arxiv_dataset/pdf_list.txt"

echo "Shard ${SHARD_ID}: Processing indices ${START}-${END}"

# 在远程机器上创建shard专用的临时目录
SHARD_TEMP_DIR="${REMOTE_TEMP_DIR}_${SHARD_ID}"
mkdir -p "${SHARD_TEMP_DIR}/pdf"
mkdir -p "${SHARD_TEMP_DIR}/md"

# 获取需要处理的PDF文件列表
python3 - <<PY > "${SHARD_TEMP_DIR}/shard_${SHARD_ID}_pdfs.txt"
from pathlib import Path
import re

# 读取本地PDF文件列表
with open('${PDF_LIST_FILE}', 'r') as f:
    all_pdfs = [line.strip() for line in f if line.strip()]

# Deduplicate by paper ID
paper_groups = {}
for pdf_path in all_pdfs:
    filename = Path(pdf_path).name
    match = re.match(r'(\d{4}\.\d{4,5})', filename)
    if match:
        paper_id = match.group(1)
        if paper_id not in paper_groups:
            paper_groups[paper_id] = []
        paper_groups[paper_id].append(pdf_path)

deduped_pdfs = []
for paper_id, versions in sorted(paper_groups.items()):
    versions.sort()
    deduped_pdfs.append(versions[0])

# Shuffle for better load balancing across shards
import random
random.seed(42)  # Fixed seed for reproducibility
random.shuffle(deduped_pdfs)

# 输出指定范围的PDF文件
for pdf in deduped_pdfs[${START}-1:${END}]:
    print(pdf)
PY

# 分批下载和处理的优化方案
echo "Starting batch download and processing..."

# 分批下载，每批100个文件
BATCH_SIZE=100
BATCH_COUNT=0

while IFS= read -r local_pdf; do
    if [[ -n "$local_pdf" ]]; then
        # 计算相对路径
        relative_path="${local_pdf#${LOCAL_PDF_DIR}}"
        relative_path="${relative_path#/}"  # 移除开头的斜杠
        remote_pdf="${SHARD_TEMP_DIR}/pdf/${relative_path}"
        
        # 确保远程目录存在
        mkdir -p "$(dirname "$remote_pdf")"
        
        # 从本地机器下载文件到远程机器
        echo "Downloading: ${LOCAL_HOST}:${local_pdf} -> ${remote_pdf}"
        scp -i "${LOCAL_SSH_KEY}" "${LOCAL_HOST}:${local_pdf}" "$remote_pdf"
        
        # 每下载100个文件就处理一批
        if (( (++BATCH_COUNT) % BATCH_SIZE == 0 )); then
            echo "Processed ${BATCH_COUNT} files, starting batch processing..."
            bash -c "source ~/.bashrc && conda activate gsam && python3 /home/suny0a/Proj/CoTT/thirdparty/2_pdf2markdown/batch_infer.py --root ${SHARD_TEMP_DIR}/pdf --outdir ${SHARD_TEMP_DIR}/md --start 1 --end 999999"
            echo "Batch processing completed, continuing download..."
        fi
    fi
done < "${SHARD_TEMP_DIR}/shard_${SHARD_ID}_pdfs.txt"

# 处理剩余的文件
if (( BATCH_COUNT % BATCH_SIZE != 0 )); then
    echo "Processing remaining ${BATCH_COUNT} files..."
    bash -c "source ~/.bashrc && conda activate gsam && python3 /home/suny0a/Proj/CoTT/thirdparty/2_pdf2markdown/batch_infer.py --root ${SHARD_TEMP_DIR}/pdf --outdir ${SHARD_TEMP_DIR}/md --start 1 --end 999999"
fi

# 将处理结果上传回本地存储机器
echo "Uploading results back to local storage..."
rsync -av -e "ssh -i ${LOCAL_SSH_KEY}" "${SHARD_TEMP_DIR}/md/" "${LOCAL_HOST}:${LOCAL_OUT_DIR}/"

# 清理远程临时目录
echo "Cleaning up remote temporary directory..."
rm -rf "${SHARD_TEMP_DIR}"

echo "Shard ${SHARD_ID} completed"
EOF

chmod +x "${PROCESS_SCRIPT}"
echo "✓ 处理脚本创建完成"

# Ensure logs directory exists for Slurm output files
echo "5. 创建日志目录..."
mkdir -p logs
echo "✓ 日志目录创建完成"

echo "6. 启动Slurm任务..."
for (( i=0; i<SHARDS; i++ )); do
  START=$(( i * PER_SHARD + 1 ))
  END=$(( (i + 1) * PER_SHARD ))
  if (( END > TOTAL )); then END=${TOTAL}; fi

  if (( START > END )); then
    echo "  Shard ${i} empty (start ${START} > end ${END}), skipping"
    continue
  fi

  echo "  启动 Shard ${i}: indices ${START}-${END}"

  srun \
    --gpus=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=64G \
    --time=96:00:00 \
    --constraint=v100 \
    --job-name=mineru_${START}_${END} \
    --output=logs/mineru_${START}_${END}.out \
    --error=logs/mineru_${START}_${END}.err \
    --unbuffered \
    bash -lc "${PROCESS_SCRIPT} ${i} ${START} ${END} '${LOCAL_HOST}' '${LOCAL_PDF_DIR}' '${LOCAL_OUT_DIR}' '${REMOTE_TEMP_DIR}'" &
done

wait
echo "✓ 所有任务已提交并完成"

# 清理本地临时文件
rm -f "${PDF_LIST_FILE}"
rm -f "${PROCESS_SCRIPT}"
rm -f "/ibex/user/suny0a/arxiv_dataset/existing_md_files.txt"
