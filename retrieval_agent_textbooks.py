"""Table 2 retrieval-agent variant using the available Textbooks corpus."""

import argparse
import json
import logging
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from liquid import Template
from tqdm import tqdm

from microservice import CustomLanguageModel
from multi_refine import FIRST_ROUND_PROMPT, REFINE_PROMPT, SYSTEM_PROMPT
from utils import (RetrievalService, calculate_accuracy, combine_docs, copy_files,
                   get_query, inference, judger)


QUERY_SYSTEM_PROMPT = '''\
### Role & Goal:
You are a medical expert. I will provide multiple different answers to the same question, which may be incorrect. Your task is to identify contradictions, ambiguities, and core dispute points among the answers, and extract key concept queries for further retrieval and verification.

### Processing Steps:
1. Analyze different answers and summarize the differences between them.
2. Extract keywords from these differences for retrieval.
3. Generate 1-4 precise queries for further search using **BM25** retriever.

### Output format:
ONLY give the queries:
[Query 1] xxx
[Query 2] xxx
(more queries)'''
QUERY_PROMPT = Template('''\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

---

I will provide several assistant's previous answers, which may be incorrect:

{{answers}}

---

Please analyze the differences among the answers and generate 4 queries for further search. Give your queries in the given format.''')
REFINE_WITH_DOCUMENTS_PROMPT = Template('''\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

### Documents
{{documents}}

### Your previous answer
{{previous_answer}}

Please critique your previous reasoning, use the relevant documents to correct any errors, and answer the question again. Give your analysis and final choice.''')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=Path, default=Path('./datasets/MMLU-PRO.json'))
    parser.add_argument('--model-name', default='qwen3-8b')
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--exp', default='retrieval_agent_textbooks_mmlu_pro_8x8')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-round', type=int, default=8)
    parser.add_argument('--start-id', type=int, default=0)
    parser.add_argument('--end-id', type=int, default=-1)
    return parser.parse_args()


def generate_queries(model, question, options, answers):
    formatted_answers = '\n\n'.join(
        f"{i}. The assistant's previous answer is:\n{answer}"
        for i, answer in enumerate(answers, start=1)
    )
    prompt = QUERY_PROMPT.render(
        question=question, options=options, answers=formatted_answers
    )
    _, content, _ = inference(
        QUERY_SYSTEM_PROMPT, prompt, model, enable_thinking=True, temperature=0.
    )[0]
    queries = re.findall(r'^\[Query\s+\d+\]\s*(.+?)\s*$', content, re.MULTILINE)
    return list(dict.fromkeys(query.strip() for query in queries if query.strip()))[:4]


def retrieve_documents(retriever, queries):
    if not queries:
        return '', 0
    retrieve = partial(retriever.retrieve, total_k=32, top_k=2,
                       combine_docs=False, use_reranker=True)
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        results = list(executor.map(retrieve, queries))
    documents = []
    seen_ids = set()
    for docs, _scores in results:
        for doc in docs:
            if doc['id'] not in seen_ids:
                seen_ids.add(doc['id'])
                documents.append(doc)
    return combine_docs(documents, combine_sep='\n'), len(documents)


def refine_path(model, question, options, previous_answer, documents):
    if documents:
        prompt = REFINE_WITH_DOCUMENTS_PROMPT.render(
            question=question, options=options,
            previous_answer=previous_answer, documents=documents
        )
    else:
        prompt = REFINE_PROMPT.render(
            question=question, options=options, previous_answer=previous_answer
        )
    return prompt, inference(SYSTEM_PROMPT, prompt, model)[0]


def main():
    args = parse_args()
    if args.num_workers < 1 or args.num_round < 1:
        raise ValueError('--num-workers and --num-round must be positive')

    log_dir = Path(
        f'runs/retrieval-agent-textbooks/{args.dataset_path.stem}/{args.model_name}/exp_{args.exp}'.lower()
    )
    result_dir = log_dir / 'evaluations'
    result_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, filename=log_dir / 'logging.log',
                        filemode='a', format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(vars(args))
    copy_files(log_dir, folders=['microservice'],
               files=['retrieval_agent_textbooks.py', 'multi_refine.py'])

    with args.dataset_path.open(encoding='utf-8') as fp:
        datasets = json.load(fp)
    end_id = args.end_id if args.end_id >= 0 else float('inf')
    datasets = [item for item in datasets
                if args.start_id <= item['id'] <= end_id
                and not (result_dir / f"question_{item['id']}").exists()]
    print(f'Left {len(datasets)}')

    model = CustomLanguageModel(args.model_name, logger, base_url=args.base_url)
    retriever = RetrievalService()
    if not retriever.host:
        raise ValueError('RETRIEVER_HOST is not set; start Textbooks retrieval and set it in .env')

    for item in tqdm(datasets, ncols=100):
        question = item['question']
        options = get_query(item)['option_str']
        prompt = FIRST_ROUND_PROMPT.render(question=question, options=options)
        paths = [(prompt, result) for result in inference(
            SYSTEM_PROMPT, prompt, model, n=args.num_workers
        )]
        rounds = []
        for round_id in range(1, args.num_round + 1):
            round_results = []
            for answer_id, (path_prompt, result) in enumerate(paths, start=1):
                reasoning, response, metadata = result
                prediction, is_correct = judger(response, item['answer'], single_answer=True)
                round_results.append(OrderedDict(
                    round=round_id, answer_id=answer_id, answer=item['answer'],
                    prediction=prediction, is_correct=is_correct, prompt=path_prompt,
                    reasoning=reasoning, response=response,
                    logprobs=metadata['logprobs'],
                    token_entropies=metadata['token_entropies'],
                ))
            rounds.append(round_results)
            if len({result['prediction'] for result in round_results}) == 1 or round_id == args.num_round:
                break

            queries = generate_queries(
                model, question, options, [result['response'] for result in round_results]
            )
            documents, document_count = retrieve_documents(retriever, queries)
            logger.info('question=%s round=%s queries=%s documents=%s',
                        item['id'], round_id, len(queries), document_count)
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                paths = list(executor.map(
                    lambda result: refine_path(
                        model, question, options, result['response'], documents
                    ),
                    round_results,
                ))

        question_dir = result_dir / f"question_{item['id']}"
        question_dir.mkdir()
        for round_id, round_results in enumerate(rounds, start=1):
            with (question_dir / f'round_{round_id}.jsonl').open('w', encoding='utf-8') as fp:
                for result in round_results:
                    fp.write(json.dumps(result, ensure_ascii=False) + '\n')

    calculate_accuracy(result_dir)


if __name__ == '__main__':
    main()
