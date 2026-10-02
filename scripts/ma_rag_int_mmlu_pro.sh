#!/usr/bin/env bash
cd /home/mmc_stu/hdd_data/lyj/project/MA_RAG || exit 1
mkdir -p logs

nohup env PYTHONUNBUFFERED=1 \
    /home/mmc_stu/anaconda3/envs/l-marag/bin/python ma_rag_entropy.py \
    --dataset-path ./datasets/MMLU-PRO.json \
    --model-name qwen3-8b \
    --exp int_mmlu_pro_8x8 \
    --num-workers 8 \
    --num-round 8 \
    > logs/ma_rag_int_mmlu_pro.log 2>&1 < /dev/null &
