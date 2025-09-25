#!/bin/bash

# One-command deployment script
# Run this on any machine to set up arXiv PDF downloader

set -e

echo "🚀 Deploying arXiv Tools (Filter + GCS Download) ..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 required. Install with:"
    echo "   Ubuntu: sudo apt install python3 python3-pip"
    echo "   macOS: brew install python3"
    exit 1
fi

echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Create quick start script (idempotent)
cat > quick_start.sh << 'QUICK_EOF'
#!/bin/bash

echo "🚀 Quick Start: arXiv Tools (Filter + GCS Download)"
echo "==================================================="

# 1) Filter metadata snapshot -> output/{paper_ids.txt, filtered_papers_metadata.xlsx}
SNAPSHOT_PATH=${SNAPSHOT_PATH:-"../../../arxiv-metadata-oai-snapshot.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output"}

echo "📊 Filtering metadata from: $SNAPSHOT_PATH"
python3 core/filter_metadata.py --snapshot "$SNAPSHOT_PATH" --output "$OUTPUT_DIR" "$@"

# 2) Check Google Cloud auth for GCS download
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "⚠️  Google Cloud not authenticated. Please run: gcloud auth login"
    echo "    Or set GOOGLE_APPLICATION_CREDENTIALS to a service account key."
    exit 1
fi
echo "✅ Google Cloud authenticated"

# 3) Download PDFs from GCS using the filtered IDs
echo "📥 Downloading PDFs from GCS using IDs in $OUTPUT_DIR/paper_ids.txt ..."
python3 core/download_papers.py --papers "$OUTPUT_DIR/paper_ids.txt" --output ./pdfs

echo "✅ Done! Artifacts:"
echo "   - IDs:        $OUTPUT_DIR/paper_ids.txt"
echo "   - Metadata:   $OUTPUT_DIR/filtered_papers_metadata.xlsx"
echo "   - PDFs dir:   ./pdfs/"
QUICK_EOF

chmod +x quick_start.sh

echo "✅ Deployment complete!"
echo ""
echo "🚀 Quick start:"
echo "   1. gcloud auth login"
echo "   2. ./quick_start.sh --year 2001 --limit 20"
echo ""
echo "📖 Or run steps manually:"
echo "   python3 core/filter_metadata.py --snapshot ../../../arxiv-metadata-oai-snapshot.json --year 2001 --output ./output"
echo "   python3 core/download_papers.py --papers ./output/paper_ids.txt --output ./pdfs"
