
# pip install -U "mineru[core]"
# pip install -U "vllm>=0.6" transformers accelerate sentencepiece xformers
mineru -p thirdparty/paper2figure-dataset/arxiv_tools/output/papers/paper2figure_dataset/pdf/0806.1636/0806.1636v1.pdf   -o outdir   --backend vlm-vllm-engine   --device cuda    --max-num-seqs 8   --max-model-len 8192

# mineru -p /path/to/dir_of_pdfs -o /path/to/outdir --backend vlm-vllm-engine --device cuda --gpu-memory-utilization 0.10 --max-num-seqs 8 --max-model-len 8192
