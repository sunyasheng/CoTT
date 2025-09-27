#!/usr/bin/env bash

# Usage:
#   bash srun_cvpr_download.sh [YEARS=13] [TIME=12:00:00] [DOWNLOADER=requests]
#   YEARS: number of years to download (2013-2025, default 13)
#   TIME: time limit per job (default 12:00:00)
#   DOWNLOADER: downloader type (requests/IDM, default requests)

set -euo pipefail

YEARS="${1:-13}"
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

echo "Starting CVPR download jobs for ${YEARS} years (2013-2025)"
echo "Time limit per job: ${TIME_LIM}"
echo "Downloader: ${DOWNLOADER}"
echo "Base save directory: ${BASE_SAVE_DIR}"

# Create download script for each year
for (( year=2013; year<=2025; year++ )); do
  if (( year > 2013 + YEARS - 1 )); then
    break
  fi
  
  echo "Submitting job for CVPR ${year}..."
  
  srun \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=8G \
    --time="$TIME_LIM" \
    --job-name=cvpr_${year} \
    --output=logs/cvpr_${year}.out \
    --error=logs/cvpr_${year}.err \
    --unbuffered \
    bash -lc "
      set -euo pipefail
      
      echo 'Starting download for CVPR ${year}...'
      echo 'Save directory: ${BASE_SAVE_DIR}/CVPR_${year}'
      echo 'Time step: ${TIME_STEP} seconds'
      
      # Activate conda environment
      echo 'Activating conda environment: scientist.sh'
      conda activate scientist.sh
      
      cd ${WORKDIR}
      
      # Create save directory
      mkdir -p ${BASE_SAVE_DIR}/CVPR_${year}
      
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
      
      if [[ '${DOWNLOADER}' == 'IDM' ]]; then
        DOWNLOADER_ARG='IDM'
      else
        DOWNLOADER_ARG='None'
      fi
      
      # Run the download
      python3 -c \"
import sys
import os
sys.path.append('${WORKDIR}')
sys.path.append('${WORKDIR}/code')

from paper_downloader_CVF import download_paper

print(f'Downloading CVPR ${year}...')
download_paper(
    year=${year},
    conference='CVPR',
    save_dir='${BASE_SAVE_DIR}/CVPR_${year}',
    is_download_main_paper=\${MAIN_BOOL},
    is_download_supplement=\${SUPP_BOOL},
    time_step_in_seconds=${TIME_STEP},
    is_download_main_conference=\${MAIN_BOOL},
    is_download_workshops=\${WORKSHOP_BOOL},
    downloader='\${DOWNLOADER_ARG}'
)
print(f'Completed CVPR ${year} download')
\"
      
      echo 'CVPR ${year} download completed successfully'
    " &
    
  # Add a small delay between job submissions
  sleep 2
done

echo "All download jobs submitted!"
echo "Monitor progress with: squeue -u \$USER"
echo "Check logs in: ${WORKDIR}/logs/"

wait
echo "All CVPR download jobs completed!"
