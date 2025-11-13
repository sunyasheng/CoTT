#!/usr/bin/env bash

# Usage:
#   bash batch_run_flist.sh
#
# Runs batch_infer_flist.py on 4 GPUs in parallel
# Each GPU processes 10,000 PDFs from the flist

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/batch_infer_flist.py"

# Configuration
GPUS=(0 1 2 3)  # GPU IDs to use
PDFS_PER_GPU=10000  # Number of PDFs each GPU processes
TOTAL_PDFS=40000  # Total: 4 GPUs × 10,000 PDFs

# Get the full path to python (to ensure we use the correct environment)
PYTHON_BIN="$(which python)"

echo "Starting batch processing on ${#GPUS[@]} GPUs"
echo "Python: ${PYTHON_BIN}"
echo "Script: ${PYTHON_SCRIPT}"
echo "Each GPU will process ${PDFS_PER_GPU} PDFs"
echo "Total: ${TOTAL_PDFS} PDFs"
echo "================================"

# Create log directory
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Launch jobs in parallel
for i in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$i]}
    START=$((i * PDFS_PER_GPU + 1))
    END=$(((i + 1) * PDFS_PER_GPU))
    
    LOG_FILE="${LOG_DIR}/gpu${GPU_ID}_${START}-${END}.log"
    
    echo "GPU ${GPU_ID}: Processing PDFs ${START} to ${END}"
    echo "  Log: ${LOG_FILE}"
    
    # Write startup info to log
    {
        echo "==================================="
        echo "Starting GPU ${GPU_ID}"
        echo "Python: ${PYTHON_BIN}"
        echo "Script: ${PYTHON_SCRIPT}"
        echo "Range: ${START} to ${END}"
        echo "CUDA_VISIBLE_DEVICES: ${GPU_ID}"
        echo "Started at: $(date)"
        echo "==================================="
    } > "${LOG_FILE}"
    
    # Run in background with specific GPU
    CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON_BIN}" -u "${PYTHON_SCRIPT}" \
        --start ${START} \
        --end ${END} \
        >> "${LOG_FILE}" 2>&1 &
    
    # Store PID
    PIDS[$i]=$!
    echo "  PID: ${PIDS[$i]}"
done

echo "================================"
echo "All jobs launched!"
echo "Monitoring progress (Ctrl+C to stop monitoring, jobs will continue)..."
echo ""

# Monitor jobs
while true; do
    ALL_DONE=true
    for i in "${!PIDS[@]}"; do
        PID=${PIDS[$i]}
        if kill -0 ${PID} 2>/dev/null; then
            ALL_DONE=false
        fi
    done
    
    if ${ALL_DONE}; then
        echo ""
        echo "All jobs completed!"
        break
    fi
    
    # Show progress every 30 seconds
    sleep 30
    echo -n "."
done

echo ""
echo "================================"
echo "Summary:"
for i in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$i]}
    START=$((i * PDFS_PER_GPU + 1))
    END=$(((i + 1) * PDFS_PER_GPU))
    LOG_FILE="${LOG_DIR}/gpu${GPU_ID}_${START}-${END}.log"
    
    # Check exit status and count successes
    if wait ${PIDS[$i]}; then
        SUCCESSES=$(grep -c "✅ SUCCESS" "${LOG_FILE}" 2>/dev/null || echo "0")
        FAILURES=$(grep -c "❌ ERROR\|⚠️  WARNING" "${LOG_FILE}" 2>/dev/null || echo "0")
        echo "GPU ${GPU_ID} (${START}-${END}): DONE - ${SUCCESSES} successes, ${FAILURES} failures"
    else
        echo "GPU ${GPU_ID} (${START}-${END}): FAILED (check log: ${LOG_FILE})"
    fi
done

echo ""
echo "View logs: ls -lh ${LOG_DIR}/"

