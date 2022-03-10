export DATA=/data/scratch/huggingface/datasets/
export TOKENIZER=/data/scratch/huggingface/tokenizers/gpt2/gpt2/
export LOG=/home/lclsg/projects/colossal/benchmark/ColossalAI-Benchmark/zero
export CUDA_VISIBLE_DEVICES=4,5

torchrun --rdzv_endpoint=localhost:29560 --nproc_per_node=2 run.py --config $1