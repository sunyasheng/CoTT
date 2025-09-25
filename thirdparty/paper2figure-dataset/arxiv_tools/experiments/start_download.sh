
## filter the relevant papers
python3 thirdparty/paper2figure-dataset/arxiv_tools/core/filter_metadata.py \
  --snapshot ./arxiv-metadata-oai-snapshot.json \
  --categories cs.CV,cs.LG,cs.AI,cs.CL,cs.HCI \
  --year 2025 \
  --limit 2000000 \
  --output thirdparty/paper2figure-dataset/arxiv_tools/output


## use public bucket to donwload
# python3 thirdparty/paper2figure-dataset/arxiv_tools/core/download_papers.py \
#     --papers thirdparty/paper2figure-dataset/arxiv_tools/output/paper_ids.txt \
#     --output thirdparty/paper2figure-dataset/arxiv_tools/pdfs \
#     --bucket arxiv-dataset \
#     --prefix arxiv/ \
#     --public
