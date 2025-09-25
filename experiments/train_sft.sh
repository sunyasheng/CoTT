mkdir -p LLaMA-Factory/data 
cp data/Ego-CoTT-25K/train-cott.json LLaMA-Factory/data/

# Train model
conda activate llamafactory
cd LLaMA-Factory
llamafactory-cli train examples/train_full/qwen.yaml
