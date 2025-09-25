#!/usr/bin/env bash

# Usage:
#   bash thirdparty/2_pdf_parsing/srun_infer.sh
#
# Splits first 20000 PDFs into 8 shards and launches 8 srun jobs.
# Each job uses 1x V100, runs up to 100 hours.

set -euo pipefail

ROOT_DIR="/ibex/user/suny0a/arxiv_dataset/pdf/"
OUT_DIR="/ibex/user/suny0a/arxiv_dataset/md/"

# Sharding config
TOTAL=20000
SHARDS=8
PER_SHARD=$(( (TOTAL + SHARDS - 1) / SHARDS ))

echo "Launching ${SHARDS} shards over first ${TOTAL} PDFs (≈${PER_SHARD}/shard)"

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
    --time=100:00:00 \
    --constraint=a100 \
    --job-name=mineru_${START}_${END} \
    --output=logs/mineru_${START}_${END}.out \
    --error=logs/mineru_${START}_${END}.err \
    --unbuffered \
    bash -lc "mkdir -p logs && conda activate gsam && export MAGIC_PDF_CONFIG=$(pwd)/thirdparty/2_pdf_parsing/magic-pdf.json && python3 thirdparty/2_pdf_parsing/batch_infer.py --root ${ROOT_DIR} --outdir ${OUT_DIR} --start ${START} --end ${END}" &
done

wait
echo "All shards submitted and completed."


