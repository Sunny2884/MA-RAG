#!/usr/bin/env bash
cd /home/mmc_stu/hdd_data/lyj/project/MA_RAG || exit 1
mkdir -p logs

nohup env PYTHONUNBUFFERED=1 \
    /home/mmc_stu/anaconda3/envs/l-marag-c/bin/python multi_refine.py \
    --dataset-path ./datasets/MMLU-PRO.json \
    --model-name qwen3-8b \
    --method qwen3-baseline \
    --exp qwen3_8b_mmlu_pro_1x1 \
    --num-workers 1 \
    --num-round 1 \
    > logs/qwen3_8b_mmlu_pro_baseline.log 2>&1 < /dev/null &
