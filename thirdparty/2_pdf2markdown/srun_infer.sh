#!/usr/bin/env bash

# Usage:
#   bash thirdparty/2_pdf_parsing/srun_infer.sh
#
# Splits all PDFs into 8 shards (shuffled for load balancing) and launches 8 srun jobs.
# Each job uses 1x A100, runs up to 48 hours.

set -euo pipefail

ROOT_DIR="/ibex/user/suny0a/arxiv_dataset/pdf/"
OUT_DIR="/ibex/user/suny0a/arxiv_dataset/md/"

# Sharding config
TOTAL=0  # 0 means process all PDFs
SHARDS=8  # Cluster limit: max 8 concurrent jobs

# Count total PDFs with deduplication
COUNT_ALL=$(python3 - <<PY
from pathlib import Path
import re
root=Path('${ROOT_DIR}').resolve()
all_pdfs = sorted(root.rglob("*.pdf"))

# Deduplicate by paper ID (same logic as batch script)
paper_groups = {}
for pdf in all_pdfs:
    filename = pdf.name
    match = re.match(r'(\d{4}\.\d{4,5})', filename)
    if match:
        paper_id = match.group(1)
        if paper_id not in paper_groups:
            paper_groups[paper_id] = []
        paper_groups[paper_id].append(pdf)

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

if [[ "$COUNT_ALL" -eq 0 ]]; then
  echo "No PDF files found under ${ROOT_DIR}" >&2
  exit 2
fi

if [[ "$TOTAL" -eq 0 ]] || [[ "$TOTAL" -gt "$COUNT_ALL" ]]; then
  TOTAL="$COUNT_ALL"
fi

PER_SHARD=$(( (TOTAL + SHARDS - 1) / SHARDS ))

# Count existing outputs
EXISTING=$(python3 - <<PY
from pathlib import Path
import re
root=Path('${ROOT_DIR}').resolve()
outdir=Path('${OUT_DIR}').resolve()
all_pdfs = sorted(root.rglob("*.pdf"))

# Deduplicate by paper ID (same logic as batch script)
paper_groups = {}
for pdf in all_pdfs:
    filename = pdf.name
    match = re.match(r'(\d{4}\.\d{4,5})', filename)
    if match:
        paper_id = match.group(1)
        if paper_id not in paper_groups:
            paper_groups[paper_id] = []
        paper_groups[paper_id].append(pdf)

deduped_pdfs = []
for paper_id, versions in sorted(paper_groups.items()):
    versions.sort()
    deduped_pdfs.append(versions[0])

# Shuffle for better load balancing across shards
import random
random.seed(42)  # Fixed seed for reproducibility
random.shuffle(deduped_pdfs)

# Check how many already have markdown output
existing = 0
for pdf in deduped_pdfs[:${TOTAL}]:
    pdf_stem = pdf.stem
    expected_md_file = outdir / pdf_stem / "vlm" / f"{pdf_stem}.md"
    if expected_md_file.exists():
        existing += 1
print(existing)
PY
)

echo "Found ${COUNT_ALL} unique papers (deduplicated)"
echo "Found ${EXISTING}/${TOTAL} PDFs already processed in ${OUT_DIR}"
echo "Launching ${SHARDS} shards over all ${TOTAL} unique papers (≈${PER_SHARD}/shard)"

# Ensure logs directory exists for Slurm output files
mkdir -p logs

for (( i=0; i<SHARDS; i++ )); do
  START=$(( i * PER_SHARD + 1 ))
  END=$(( (i + 1) * PER_SHARD ))
  if (( END > TOTAL )); then END=${TOTAL}; fi

  if (( START > END )); then
    echo "Shard ${i} empty (start ${START} > end ${END}), skipping"
    continue
  fi

  echo "Shard ${i}: indices ${START}-${END}"

  srun \
    --gpus=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=64G \
    --time=48:00:00 \
    --constraint=a100 \
    --job-name=mineru_${START}_${END} \
    --output=logs/mineru_${START}_${END}.out \
    --error=logs/mineru_${START}_${END}.err \
    --unbuffered \
    bash -lc "mkdir -p logs && conda activate gsam && python3 thirdparty/2_pdf_parsing/batch_infer.py --root ${ROOT_DIR} --outdir ${OUT_DIR} --start ${START} --end ${END}" &
done

wait
echo "All shards submitted and completed."


