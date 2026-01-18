import json
import argparse
from src.seren_uplift import SerenUplift


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--top_k', type=int, default=20,
        help='Number of top recommendations to generate'
    )
    parser.add_argument(
        '--uplift_result_path', type=str,
        default='data/processed_data/pred_uplift.csv',
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

    return parser.parse_args()


def main():
    args = parse_args()
    configs = vars(args)

    model = SerenUplift()
    save_path = 'results/recommendation/SerenUplift.json'

    recommendations = model.recommend(**configs)

    with open(save_path, mode='w') as f:
        json.dump(
            recommendations, f, indent=4
        )


if __name__ == "__main__":
    main()
