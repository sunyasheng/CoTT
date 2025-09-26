#!/usr/bin/env bash

# Usage (directory + sharding by indices):
#   bash thirdparty/3_roi/srun_qwen_select_and_caption.sh /abs/path/to/markdown_root_dir [SHARDS=8] [TIME=48:00:00] [GLOB='*/vlm/*.md']

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <md_root_dir> [SHARDS=8] [TIME=48:00:00] [GLOB='*/vlm/*.md']" >&2
  exit 2
fi

ROOT_DIR="$1"
SHARDS="${2:-8}"
TIME_LIM="${3:-48:00:00}"
GLOB="${4:-*/vlm/*.md}"
# Optional: limit total processed items (like srun_infer.sh). If unset, use full count.
TOTAL_LIMIT="${TOTAL:-}"  # can set via env TOTAL=20000

MODEL=${MODEL:-"Qwen/Qwen2.5-VL-32B-Instruct"}
MAX_FIGURES=${MAX_FIGURES:-3}
MAX_CAPTIONS=${MAX_CAPTIONS:-0}
TIMEOUT=${TIMEOUT:-120}
OUTDIR=${OUTDIR:-"/ibex/user/suny0a/arxiv_dataset/thinking"}

QWEN_BASE_URL=${QWEN_BASE_URL:-}
QWEN_API_KEY=${QWEN_API_KEY:-EMPTY}

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="${WORKDIR}/batch_qwen_select_and_caption.py"

mkdir -p logs

COUNT_ALL=$(python3 - <<PY
from pathlib import Path
root=Path('${ROOT_DIR}').resolve()
mds=sorted(root.rglob('${GLOB}'))
print(len(mds))
PY
)

if [[ "$COUNT_ALL" -eq 0 ]]; then
  echo "No markdown files found under ${ROOT_DIR} with glob ${GLOB}" >&2
  exit 2
fi

if [[ -n "$TOTAL_LIMIT" ]]; then
  if [[ "$TOTAL_LIMIT" -gt "$COUNT_ALL" ]]; then
    TOTAL="$COUNT_ALL"
  else
    TOTAL="$TOTAL_LIMIT"
  fi
else
  TOTAL="$COUNT_ALL"
fi

PER_SHARD=$(( (TOTAL + SHARDS - 1) / SHARDS ))
echo "Launching ${SHARDS} shards over ${TOTAL}/${COUNT_ALL} markdowns (≈${PER_SHARD}/shard)"

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
    --mem=32G \
    --time="$TIME_LIM" \
    --constraint=a100 \
    --job-name=qwen_fig_${START}_${END} \
    --output=logs/qwen_fig_${START}_${END}.out \
    --error=logs/qwen_fig_${START}_${END}.err \
    --unbuffered \
    bash -lc '
      set -euo pipefail
      PORT=$(python3 - <<PY
import socket
s=socket.socket()
s.bind(("",0))
print(s.getsockname()[1])
s.close()
PY
)
      echo "Starting Qwen server on port ${PORT}..."
      conda activate gsam
      bash "${WORKDIR}/qwen_server_setup.sh" --port "${PORT}" > "logs/qwen_server_${START}_${END}.out" 2>&1 &
      for i in $(seq 1 60); do
        if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
          echo "Qwen server is up on 127.0.0.1:${PORT}"
          break
        fi
        echo "Waiting for Qwen server (attempt $i)..."; sleep 5
      done
      export QWEN_API_KEY="${QWEN_API_KEY}"
      export QWEN_BASE_URL="http://127.0.0.1:${PORT}/v1"
      python3 "${RUNNER}" --root "${ROOT_DIR}" --start ${START} --end ${END} --glob "${GLOB}" --model "${MODEL}" --max_figures ${MAX_FIGURES} --max_captions ${MAX_CAPTIONS} --timeout ${TIMEOUT} --outdir "${OUTDIR}"
    ' &
done

wait
echo "All shards submitted and completed."


# TOTAL=20000 bash thirdparty/3_roi/srun_qwen_select_and_caption.sh \
#   /ibex/user/suny0a/arxiv_dataset/md \
#   8 \
#   48:00:00 \
#   "*/vlm/*.md"