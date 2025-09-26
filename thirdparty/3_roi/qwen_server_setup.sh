#!/usr/bin/env bash
set -euo pipefail

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      VLLM_PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# 1) Activate conda env (robust, non-interactive)
PYTHON=python
if [ "${CONDA_DEFAULT_ENV-}" != "gsam" ]; then
  # try common init methods
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" || true
  fi
  if [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    . "$HOME/.conda/etc/profile.d/conda.sh" || true
  fi
  if [ -f "/etc/profile.d/conda.sh" ]; then
    . "/etc/profile.d/conda.sh" || true
  fi
  if [ -x "$HOME/miniconda3/bin/conda" ]; then
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)" || true
  fi

  if command -v conda >/dev/null 2>&1; then
    conda activate gsam || true
  fi

  # If still not active, fallback to conda run
  if [ "${CONDA_DEFAULT_ENV-}" != "gsam" ]; then
    echo "[INFO] Falling back to 'conda run -n gsam' without activate."
    PYTHON="conda run -n gsam --no-capture-output python"
  else
    PYTHON=python
  fi
fi

# 2) Configure caches to non-home large disk
export HF_HOME=/ibex/user/suny0a/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export VLLM_CACHE_DIR=/ibex/user/suny0a/vllm_cache
# Triton autotune cache (avoid NFS)
export TRITON_CACHE_DIR=/ibex/user/suny0a/triton_autotune
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$VLLM_CACHE_DIR"
mkdir -p "$TRITON_CACHE_DIR"

# 3) Ensure dependencies (idempotent)
$PYTHON - <<'PY'
import importlib, sys
missing = []
for pkg in ["vllm", "transformers", "qwen_vl_utils"]:
    try:
        importlib.import_module(pkg)
    except Exception:
        missing.append(pkg)
if missing:
    print(f"Installing: {missing}")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *missing])
else:
    print("Python deps OK")
PY

# 4) Launch OpenAI-compatible server (single A100 80G defaults, override via env)
VLLM_PORT=${VLLM_PORT:-22002}
VLLM_MODEL=${VLLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}
VLLM_MAX_LEN=${VLLM_MAX_LEN:-12288}
VLLM_NUM_SEQS=${VLLM_NUM_SEQS:-4}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.92}

echo "[INFO] Starting vLLM server on port ${VLLM_PORT} with model ${VLLM_MODEL}"
$PYTHON -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL}" \
  --trust-remote-code \
  --download-dir "$HF_HOME" \
  --max-model-len "${VLLM_MAX_LEN}" \
  --max-num-seqs "${VLLM_NUM_SEQS}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
  --port "${VLLM_PORT}"

echo "[INFO] Server started. Example client env:"
echo "export QWEN_BASE_URL=http://127.0.0.1:${VLLM_PORT}/v1"
echo "export QWEN_API_KEY=EMPTY"
