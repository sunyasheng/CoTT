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
