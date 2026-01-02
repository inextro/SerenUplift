import os
import argparse
import torch
from recbole.config import Config
from src.recbole import RecboleRunner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, required=True,
        choices=['LightGCN', 'NeuMF', 'GRU4Rec', 'SASRec', 'BERT4Rec'],
        help='Model name to train/inference'
    )
    parser.add_argument(
        '--config', type=str, default='src/recbole_configs/overall.yaml',
        help='Path to base configuration file'
    )
    parser.add_argument(
        '--top_k', type=int, default=20,
        help='Number of top recommendations to generate'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )
    parser.add_argument(
        '--save_path', type=str, default='results/recommendation/',
        help='Path to recommendation results. If not provided, defaults to results/recommendation/{model}.json'
    )
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='Path to checkpoint file. If provided, do not train the model and only generate recommendations'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_config_path = f'src/recbole_configs/{args.model}.yaml'
    config_file_list = [args.config, model_config_path]

    config = Config(
        model=args.model, dataset='ml-belief', config_file_list=config_file_list
    )
    print(config)

    runner = RecboleRunner(config)

    if args.ckpt_path:
        runner.load_model(args.ckpt_path)
    else:
        runner.train()

    save_path = os.path.join(args.save_path, f'{args.model}.json')
    runner.inference(
        top_k=args.top_k, device=args.device, save_path=save_path
    )


if __name__ == '__main__':
    main()
