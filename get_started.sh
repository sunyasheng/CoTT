conda create -n pdf_parsing python==3.10 -y
eval "$(conda shell.bash hook)"
conda activate pdf_parsing
which python

pip install -r scripts/pdf2markdown/requirements_parsing.txt
