#!/bin/bash

# WACV Paper Download Script using srun
# Usage: bash srun_wacv_download.sh [YEARS=6] [TIME=12:00:00] [DOWNLOADER=requests]
#   YEARS: number of years to download (2020-2025, default 6)
#   TIME: time limit per job (default 12:00:00)
#   DOWNLOADER: downloader type (requests/IDM, default requests)

# set -euo pipefail  # Temporarily disabled to debug srun issue

YEARS="${1:-6}"
TIME_LIM="${2:-12:00:00}"
DOWNLOADER="${3:-requests}"

BASE_SAVE_DIR="/ibex/user/suny0a/cvf_dataset/pdf"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"

# Download configuration
TIME_STEP="${TIME_STEP:-5}"  # seconds between downloads
DOWNLOAD_MAIN="${DOWNLOAD_MAIN:-true}"
DOWNLOAD_WORKSHOPS="${DOWNLOAD_WORKSHOPS:-true}"
DOWNLOAD_SUPPLEMENT="${DOWNLOAD_SUPPLEMENT:-false}"

mkdir -p logs

echo "Starting WACV download jobs for ${YEARS} years (2020-2025)"
echo "Time limit per job: ${TIME_LIM}"
echo "Downloader: ${DOWNLOADER}"
echo "Base save directory: ${BASE_SAVE_DIR}"

# Create download script for each year
for (( year=2020; year<=2025; year++ )); do
  if (( year > 2020 + YEARS - 1 )); then
    break
  fi
  
  echo "Submitting job for WACV ${year}..."
  echo "DEBUG: About to run srun command for year ${year}"
  
  srun \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=8G \
    --time="$TIME_LIM" \
    --job-name=wacv_${year} \
    --output=logs/wacv_${year}.out \
    --error=logs/wacv_${year}.err \
    --unbuffered \
    bash -lc "
      set -euo pipefail
      
      echo 'Starting download for WACV ${year}...'
      echo 'Save directory: ${BASE_SAVE_DIR}/WACV_${year}'
      echo 'Time step: ${TIME_STEP} seconds'
      
      # Activate conda environment
      echo 'Activating conda environment: scientist.sh'
      conda activate scientist.sh
      
      cd ${WORKDIR}
      
      # Create save directory
      mkdir -p ${BASE_SAVE_DIR}/WACV_${year}
      
      # Convert boolean strings to Python booleans
      if [[ '${DOWNLOAD_MAIN}' == 'true' ]]; then
        MAIN_BOOL='True'
      else
        MAIN_BOOL='False'
      fi
      
      if [[ '${DOWNLOAD_WORKSHOPS}' == 'true' ]]; then
        WORKSHOP_BOOL='True'
      else
        WORKSHOP_BOOL='False'
      fi
      
      if [[ '${DOWNLOAD_SUPPLEMENT}' == 'true' ]]; then
        SUPP_BOOL='True'
      else
        SUPP_BOOL='False'
      fi
      
      # Run the download
      python3 -c \"
import sys
import os
sys.path.append('${WORKDIR}')
sys.path.append('${WORKDIR}/code')

from paper_downloader_CVF import download_paper

print(f'Downloading WACV ${year}...')
download_paper(
    year=${year},
    conference='WACV',
    save_dir='${BASE_SAVE_DIR}/WACV_${year}',
    is_download_main_paper=\${MAIN_BOOL},
    is_download_supplement=\${SUPP_BOOL},
    time_step_in_seconds=${TIME_STEP},
    is_download_main_conference=\${MAIN_BOOL},
    is_download_workshops=\${WORKSHOP_BOOL},
    downloader=None
)
print(f'Completed WACV ${year} download')
\"
      
      echo 'WACV ${year} download completed successfully'
    " &
    
  echo "DEBUG: srun command submitted for year ${year}"
  
  # Add a small delay between job submissions
  sleep 2
done

echo "All WACV download jobs submitted!"
echo "Monitor progress with: squeue -u \$USER"
echo "Check logs in: ${WORKDIR}/logs/"

wait
echo "All WACV download jobs completed!"
