#!/usr/bin/env bash

# ICCV paper download script
# Usage: bash srun_iccv_download.sh [YEARS=6] [TIME=12:00:00]

set -euo pipefail

YEARS="${1:-6}"  # Number of ICCV years to download (2013,2015,2017,2019,2021,2023)
TIME_LIM="${2:-12:00:00}"
DOWNLOADER="${3:-requests}"

BASE_SAVE_DIR="/ibex/user/suny0a/cvf_dataset/pdf"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"

# Download configuration
TIME_STEP="${TIME_STEP:-5}"
DOWNLOAD_MAIN="${DOWNLOAD_MAIN:-true}"
DOWNLOAD_WORKSHOPS="${DOWNLOAD_WORKSHOPS:-true}"
DOWNLOAD_SUPPLEMENT="${DOWNLOAD_SUPPLEMENT:-false}"

mkdir -p logs

echo "Starting ICCV download jobs for ${YEARS} years"
echo "Time limit per job: ${TIME_LIM}"
echo "Downloader: ${DOWNLOADER}"
echo "Base save directory: ${BASE_SAVE_DIR}"

# ICCV available years (odd years only)
ICCV_YEARS=(2013 2015 2017 2019 2021 2023)

# Create download script for each year
count=0
for year in "${ICCV_YEARS[@]}"; do
  # Check if we've reached the limit
  if (( count >= YEARS )); then
    break
  fi
  ((count++))
  
  echo "Submitting job for ICCV ${year}..."
  
  srun \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=8G \
    --time="$TIME_LIM" \
    --job-name=iccv_${year} \
    --output=logs/iccv_${year}.out \
    --error=logs/iccv_${year}.err \
    --unbuffered \
    bash -lc "
      set -euo pipefail
      
      echo 'Starting download for ICCV ${year}...'
      echo 'Save directory: ${BASE_SAVE_DIR}/ICCV_${year}'
      echo 'Time step: ${TIME_STEP} seconds'
      
      # Activate conda environment
      echo 'Activating conda environment: scientist.sh'
      conda activate scientist.sh
      
      cd ${WORKDIR}
      
      # Create save directory
      mkdir -p ${BASE_SAVE_DIR}/ICCV_${year}
      
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

print(f'Downloading ICCV ${year}...')
download_paper(
    year=${year},
    conference='ICCV',
    save_dir='${BASE_SAVE_DIR}/ICCV_${year}',
    is_download_main_paper=\${MAIN_BOOL},
    is_download_supplement=\${SUPP_BOOL},
    time_step_in_seconds=${TIME_STEP},
    is_download_main_conference=\${MAIN_BOOL},
    is_download_workshops=\${WORKSHOP_BOOL},
    downloader=None
)
print(f'Completed ICCV ${year} download')
\"
      
      echo 'ICCV ${year} download completed successfully'
    " &
    
  # Add a small delay between job submissions
  sleep 2
done

echo "All ICCV download jobs submitted!"
echo "Monitor progress with: squeue -u \$USER"
echo "Check logs in: ${WORKDIR}/logs/"

wait
echo "All ICCV download jobs completed!"
