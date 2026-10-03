import argparse
import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from liquid import Template
from tqdm import tqdm

from microservice import CustomLanguageModel
from utils import calculate_accuracy, copy_files, get_query, inference, judger


SYSTEM_PROMPT = '''You are a medical assistant, please answer my medical questions. Give your final choice (capital option) closed in tag `<answer>X</answer>` after your analysis.
Your response should be as detailed as possible, but please do not use any subheadings.'''
FIRST_ROUND_PROMPT = Template('''\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

Please analyze the question and give your answer. Give your analysis and final choice.''')
REFINE_PROMPT = Template('''\
Below is a multiple-choice question.
### Question
{{question}}

### Options
{{options}}

### Your previous answer
{{previous_answer}}

Please critique your previous reasoning, correct any errors, and answer the question again. Give your analysis and final choice.''')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=Path, default=Path('./datasets/MMLU-PRO.json'))
    parser.add_argument('--model-name', default='qwen3-8b')
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--exp', default='multi_refine_mmlu_pro_8x8')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-round', type=int, default=8)
    parser.add_argument('--start-id', type=int, default=0)
    parser.add_argument('--end-id', type=int, default=-1)
    return parser.parse_args()


def refine_path(model, question, options, previous_answer):
    prompt = REFINE_PROMPT.render(
        question=question, options=options, previous_answer=previous_answer
    )
    return prompt, inference(SYSTEM_PROMPT, prompt, model)[0]


def main():
    args = parse_args()
    if args.num_workers < 1 or args.num_round < 1:
        raise ValueError('--num-workers and --num-round must be positive')

    log_dir = Path(f'runs/multi-refine/{args.dataset_path.stem}/{args.model_name}/exp_{args.exp}'.lower())
    result_dir = log_dir / 'evaluations'
    result_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, filename=log_dir / 'logging.log',
                        filemode='a', format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(vars(args))
    copy_files(log_dir, folders=['microservice'], files=['multi_refine.py'])

    with args.dataset_path.open(encoding='utf-8') as fp:
        datasets = json.load(fp)
    end_id = args.end_id if args.end_id >= 0 else float('inf')
    datasets = [item for item in datasets
                if args.start_id <= item['id'] <= end_id
                and not (result_dir / f"question_{item['id']}").exists()]
    print(f'Left {len(datasets)}')

    model = CustomLanguageModel(args.model_name, logger, base_url=args.base_url)
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

            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                paths = list(executor.map(
                    lambda result: refine_path(model, question, options, result['response']),
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
