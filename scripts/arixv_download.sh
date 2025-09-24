#!/usr/bin/env bash
set -euo pipefail

# Descending yearly download for arXiv CS papers.
# Customize via env vars before running, e.g.:
#   START_YEAR=2025 END_YEAR=2018 MAX_BYTES=200000000000 CONCURRENCY=6 DOWNLOAD_DELAY=0.4 bash scripts/arixv_download.sh

START_YEAR="${START_YEAR:-2025}"
END_YEAR="${END_YEAR:-2015}"
MAX_BYTES="${MAX_BYTES:-1000000000}"        # default ~1GB per year; change as needed
CONCURRENCY="${CONCURRENCY:-6}"
DOWNLOAD_DELAY="${DOWNLOAD_DELAY:-0.3}"
OUT_DIR_ROOT="${OUT_DIR_ROOT:-./arxiv_pdfs}"

echo "Downloading arXiv CS PDFs from ${START_YEAR} down to ${END_YEAR}..."

for (( Y=START_YEAR; Y>=END_YEAR; Y-- )); do
  OUT_DIR="${OUT_DIR_ROOT}/${Y}"
  mkdir -p "${OUT_DIR}"
  echo "Year ${Y} -> ${OUT_DIR} (limit ${MAX_BYTES} bytes)"
  python scripts/data_crawler.py \
    --query "cat:cs.*" \
    --from-date "${Y}-01-01" \
    --until-date "${Y}-12-31" \
    --max-total-bytes "${MAX_BYTES}" \
    --out-dir "${OUT_DIR}" \
    --concurrency "${CONCURRENCY}" \
    --download-delay "${DOWNLOAD_DELAY}" \
    --resume
done

echo "Done."