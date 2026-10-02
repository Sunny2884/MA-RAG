#!/usr/bin/env bash
cd /home/mmc_stu/hdd_data/lyj/project/MA_RAG || exit 1
source /home/mmc_stu/anaconda3/etc/profile.d/conda.sh || exit 1
conda activate l-marag-c || exit 1
mkdir -p logs

nohup env CUDA_VISIBLE_DEVICES=1 python microservice/RetrievalSystem.py \
    --retriever BM25 \
    --reranker MedCPT-Cross-Encoder \
    --corpus-name Textbooks \
    --cuda 0 \
    --port 8990 \
    > logs/retrieval_textbooks.log 2>&1 < /dev/null &
