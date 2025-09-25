
## filter the relevant papers
# python3 thirdparty/paper2figure-dataset/arxiv_tools/core/filter_metadata.py \
# python3 thirdparty/paper2figure-dataset/arxiv_tools/dataset_pipeline/filter_papers.py \
#   -s cs.CV,cs.LG,cs.AI,cs.CL,cs.HCI \
#   -y 20 -m 01 \
#   -p ./arxiv-metadata-oai-snapshot.json


## use public bucket to donwload
python3 thirdparty/paper2figure-dataset/arxiv_tools/dataset_pipeline/download_papers.py \
  -p sage-now-232008 \
  --paper_ids thirdparty/paper2figure-dataset/arxiv_tools/output/paper_ids.txt \
  -out thirdparty/paper2figure-dataset/arxiv_tools/output/papers
