# arXiv Tools - Organized

Minimal, two-feature toolkit:
- Filter the `arxiv-metadata-oai-snapshot.json` to produce paper IDs and a metadata table
- Download the corresponding PDFs from Google Cloud Storage (GCS)

## Quick Deploy (Any Machine)

```bash
# Clone or download this directory
cd organized_arxiv_tools

# One-command setup
./deploy.sh

# Quick start
gcloud auth login
./quick_start.sh --year 2001 --limit 50
```

## Manual Setup

```bash
# Install dependencies
pip3 install -r requirements.txt

# Step 1: Filter snapshot -> output files
python3 core/filter_metadata.py \
  --snapshot ../../../arxiv-metadata-oai-snapshot.json \
  --year 2001 \
  --output ./output

# Step 2: Download PDFs from GCS using filtered IDs
gcloud auth login
python3 core/download_papers.py --papers ./output/paper_ids.txt --output ./pdfs
```

## Usage Examples

```bash
# Filter controls
python3 core/filter_metadata.py --snapshot ../../../arxiv-metadata-oai-snapshot.json --categories cs.CL,cs.CV --limit 100 --output ./output

# Download specific papers from GCS
python3 core/download_papers.py --papers "2001.00081,2001.00012,2001.00003" --output ./pdfs

# Download from file
python3 core/download_papers.py --papers ./output/paper_ids.txt --output ./pdfs

# Batch download by year
./examples/batch_download.sh 2001 2002
```

## File Structure

```
organized_arxiv_tools/
├── core/
│   ├── filter_metadata.py    # Filter snapshot -> IDs + metadata
│   └── download_papers.py    # Download PDFs from GCS using IDs
├── requirements.txt          # Python dependencies
├── deploy.sh                 # One-command deployment
├── quick_start.sh            # Quick start script (created by deploy.sh)
└── README.md                 # This file
```

## Requirements

- Python 3.8+
- Google Cloud Storage access
- Internet connection

## Authentication

```bash
# Option 1: gcloud CLI
gcloud auth login

# Option 2: Service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```
