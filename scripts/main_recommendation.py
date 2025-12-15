import json
import argparse
from src.rec_models.lightgcn import LightGCN
from src.rec_models.seren_uplift import SerenUplift

MODEL_DICT = {
    'seren_uplift': SerenUplift,
    'lightgcn': LightGCN
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, required=True, choices=MODEL_DICT.keys(),
        help='Recommender model to use'
    )
    parser.add_argument(
        '--top_k', type=int, default=20,
        help='Number of top recommendations to generate'
    )
    # SerenUplift
    parser.add_argument(
        '--uplift_result_path', type=str,
        default='./data/processed_data/pred_uplift.csv',
        help='Path to the uplift result file'
    )
    parser.add_argument(
        '--uplift_threshold', type=float, default=2.0,
        help='Uplift threshold (default: 2.0)'
    )
    parser.add_argument(
        '--post_threshold', type=float, default=4.0,
        help='Predicted post rating threshold (default: 4.0)'
    )
    # Baselines
    parser.add_argument(
        '--checkpoint_path', type=str,
        help='Path to the checkpoint'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    configs = vars(args)

    if configs['model'] == 'seren_uplift':
        model = SerenUplift()
        save_path = (
            f'./results/{configs['model']}_{configs['uplift_threshold']}'
            f'_{configs['post_threshold']}.json'
        )
    if configs['model'] == 'lightgcn':
        model = LightGCN()
        save_path = f'./results/{configs['model']}.json'

    recommendations = model.recommend(**configs)

    with open(save_path, mode='w') as f:
        json.dump(
            recommendations, f, indent=4
        )


if __name__ == "__main__":
    main()
