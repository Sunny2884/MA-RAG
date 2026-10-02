#!/usr/bin/env bash
source /home/mmc_stu/anaconda3/etc/profile.d/conda.sh || exit 1
conda activate l-qwen3-vllm || exit 1
mkdir -p /home/mmc_stu/hdd_data/lyj/project/MA_RAG/logs

nohup env CUDA_VISIBLE_DEVICES=6,7 \
    vllm serve \
    /home/mmc_stu/hdd_data/lyj/llm_weights/Qwen/Qwen3-8B \
    --served-model-name qwen3-8b \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.85 \
    --api-key dummy \
    --host 127.0.0.1 \
    --port 8000 \
    > /home/mmc_stu/hdd_data/lyj/project/MA_RAG/logs/qwen3_8b.log 2>&1 < /dev/null &
