# Prepare data
mkdir -p Ego-R1-Agent/data
cp data/Ego-CoTT-raw/*.parquet Ego-R1-Agent/data/

# Start RL training
conda activate egor1
cd Ego-R1-Agent
bash train_grpo.sh  # For GRPO training
