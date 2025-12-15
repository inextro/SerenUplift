import os
import json
import argparse
from dotenv import load_dotenv
from src.evaluate import SerendipityEvaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rec_path', type=str, required=True,
        help='Path to the recommendation list'
    )
    parser.add_argument(
        '--base_output_path', type=str, default='results/evaluation',
        help='Base output path for the evaluaton results'
    )
    parser.add_argument(
        '--user_map_path', type=str, default='lgcn/data/movielens/user_list.txt',
        help='Path to user mapping file'
    )
    parser.add_argument(
        '--movie_map_path', type=str, default='lgcn/data/movielens/item_list.txt',
        help='Path to movie mapping file'
    )
    parser.add_argument(
        '--movie_metadata_path', type=str, default='data/raw_data/movies.csv',
        help='Path to movie metadata file'
    )
    parser.add_argument(
        '--prompt_path', type=str, default='src/prompts/zero_shot.yaml',
        help='Path to the pre-defined prompt file'
    )
    parser.add_argument(
        '--user_history_path', type=str, required=True,
        help='Path to user rating history file'
    )
    parser.add_argument(
        '--llm', type=str, default='gpt-5-nano',
        help='Model to use for evaluation'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='If true, do not call the API'
    )

    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    rec_filename = os.path.basename(args.rec_path)
    save_filename = f'{os.path.splitext(rec_filename)[0]}.json'
    save_path = os.path.join(args.base_output_path, save_filename)

    print(f'Evaluation target: {args.rec_path}')
    print(f'LLM: {args.llm}')
    print(f'Evaluation results will be saved to: {save_path}')

    evaluator = SerendipityEvaluator(
        user_map_path=args.user_map_path,
        movie_map_path=args.movie_map_path,
        movie_metadata_path=args.movie_metadata_path,
        prompt_path=args.prompt_path,
        api_key=os.getenv('OPENAI_API_KEY'),
        llm=args.llm,
    )

    with open(args.rec_path, 'r') as f:
        recommendations = json.load(f)

    evaluator.evaluate(
        recommendations=recommendations,
        user_history_path=args.user_history_path,
        output_path=save_path,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
