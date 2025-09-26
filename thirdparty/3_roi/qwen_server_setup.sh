#!/usr/bin/env bash
set -euo pipefail

# 1) Activate conda env
# Adjust the path to your conda.sh if different on this cluster
if [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
  . "$HOME/.conda/etc/profile.d/conda.sh"
elif [ -f "/etc/profile.d/conda.sh" ]; then
  . "/etc/profile.d/conda.sh"
fi
conda activate gsam

# 2) Configure caches to non-home large disk
export HF_HOME=/ibex/user/suny0a/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export VLLM_CACHE_DIR=/ibex/user/suny0a/vllm_cache
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$VLLM_CACHE_DIR"

# 3) Ensure dependencies (idempotent)
python - <<'PY'
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

# 4) Launch OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --trust-remote-code \
  --download-dir "$HF_HOME" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 22002
