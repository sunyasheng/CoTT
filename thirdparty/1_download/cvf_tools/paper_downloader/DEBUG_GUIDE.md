# CVPR Download Debug Guide

## How to Debug on a Single Node

### 1. Get a Compute Node
```bash
# Request an interactive node
srun --ntasks=1 --cpus-per-task=4 --mem=8G --time=2:00:00 --pty bash

# Or request a specific partition if needed
srun --ntasks=1 --cpus-per-task=4 --mem=8G --time=2:00:00 --partition=cpu --pty bash
```

### 2. Navigate to the Download Directory
```bash
cd /path/to/your/paper_downloader
```

### 3. Run Debug Script

#### Basic Debug (Test CSV generation only)
```bash
python3 debug_single_year.py --year 2024 --test-only
```

#### Full Debug (Download one year)
```bash
python3 debug_single_year.py --year 2024
```

#### Debug with Custom Settings
```bash
python3 debug_single_year.py \
  --year 2024 \
  --save-dir /ibex/user/suny0a/cvf_dataset/pdf \
  --download-main \
  --download-workshops \
  --time-step 3
```

#### Debug Recent Year (Recommended for testing)
```bash
python3 debug_single_year.py --year 2024 --time-step 2
```

## Debug Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--year` | Year to download | Required | `--year 2024` |
| `--save-dir` | Save directory | `/ibex/user/suny0a/cvf_dataset/pdf` | `--save-dir /tmp/test` |
| `--download-main` | Download main conference | True | `--download-main` |
| `--download-workshops` | Download workshops | True | `--download-workshops` |
| `--download-supplement` | Download supplement | False | `--download-supplement` |
| `--time-step` | Download interval (seconds) | 5 | `--time-step 2` |
| `--downloader` | Downloader type | requests | `--downloader requests` |
| `--test-only` | Only test CSV generation | False | `--test-only` |

## Step-by-Step Debug Process

### Step 1: Test CSV Generation
```bash
# Test if we can get paper URLs
python3 debug_single_year.py --year 2024 --test-only
```

Expected output:
```
✅ Main conference: 2716 papers found
✅ Workshops: 773 papers found
Test completed successfully!
```

### Step 2: Test Download (Small Scale)
```bash
# Download with fast interval to test quickly
python3 debug_single_year.py --year 2024 --time-step 1
```

### Step 3: Check Results
```bash
# Check downloaded files
ls -la /ibex/user/suny0a/cvf_dataset/pdf/CVPR_2024/
ls -la /ibex/user/suny0a/cvf_dataset/pdf/CVPR_2024/CVPR_2024/main_paper/ | head -10
```

### Step 4: Monitor Progress
```bash
# Watch download progress in real-time
watch -n 5 'ls /ibex/user/suny0a/cvf_dataset/pdf/CVPR_2024/CVPR_2024/main_paper/ | wc -l'
```

## Common Issues and Solutions

### Issue 1: CSV Generation Fails
```bash
# Check network connectivity
curl -I "https://openaccess.thecvf.com/CVPR2024"

# Check if year exists
python3 debug_single_year.py --year 2023 --test-only
```

### Issue 2: Download Fails
```bash
# Check disk space
df -h /ibex/user/suny0a/cvf_dataset/pdf

# Check permissions
ls -la /ibex/user/suny0a/cvf_dataset/pdf
```

### Issue 3: Slow Download
```bash
# Reduce time step for faster download
python3 debug_single_year.py --year 2024 --time-step 2

# Or download only main conference first
python3 debug_single_year.py --year 2024 --download-workshops=False
```

## Recommended Debug Sequence

1. **Start with test-only mode:**
   ```bash
   python3 debug_single_year.py --year 2024 --test-only
   ```

2. **If test passes, try small download:**
   ```bash
   python3 debug_single_year.py --year 2024 --time-step 2
   ```

3. **Monitor progress:**
   ```bash
   watch -n 10 'find /ibex/user/suny0a/cvf_dataset/pdf/CVPR_2024 -name "*.pdf" | wc -l'
   ```

4. **If successful, scale up:**
   ```bash
   # Run full batch download
   python3 batch_download_cvpr.py --start-year 2020 --end-year 2025
   ```

## Quick Commands for Different Scenarios

```bash
# Quick test (2024 only, fast)
python3 debug_single_year.py --year 2024 --time-step 2

# Test older year (2020)
python3 debug_single_year.py --year 2020 --test-only

# Test with custom save directory
python3 debug_single_year.py --year 2024 --save-dir /tmp/cvpr_test

# Test only main conference (no workshops)
python3 debug_single_year.py --year 2024 --download-workshops=False
```
