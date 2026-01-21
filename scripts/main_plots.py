import os
import json
import numpy as np
from matplotlib import pyplot as plt


def load_serendipity_scores(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    scores = []

    for user_id, recommendations in data.items():
        for recommendation in recommendations:
            scores.append(int(recommendation['serendipity_rating']))

    return scores


def main(evaluation_dir):
    seren_uplift_path = os.path.join(evaluation_dir, 'SerenUplift.json')
    seren_uplift_scores = load_serendipity_scores(seren_uplift_path)

    baselines = ['NeuMF', 'LightGCN', 'GRU4Rec', 'SASRec', 'BERT4Rec']

    fig, axes = plt.subplots(1, len(baselines), figsize=(5 * len(baselines), 5))
    bins = np.arange(0.5, 6.5, 1)

    for i, baseline in enumerate(baselines):
        baseline_path = os.path.join(evaluation_dir, f'{baseline}.json')
        baseline_scores = load_serendipity_scores(baseline_path)

        ax = axes[i]

        ax.hist(seren_uplift_scores, bins=bins, alpha=0.5, label='SerenUplift', density=True, edgecolor='black', color='orange')
        ax.hist(baseline_scores, bins=bins, alpha=0.5, label=baseline, density=True, edgecolor='black', color='skyblue')

        ax.set_title(f'SerenUplift vs. {baseline}')
        ax.set_xlabel('Serendipity Rating')
        ax.set_ylabel('Frequency')
        ax.set_xticks(range(1, 6))
        ax.set_yticks(np.arange(0, 0.4, 0.05))
        ax.legend()

    plt.tight_layout()
    plt.savefig('results/assets/score_distributions.png')


if __name__ == '__main__':
    main(evaluation_dir='results/evaluation/')
