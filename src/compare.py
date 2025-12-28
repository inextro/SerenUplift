import json
import numpy as np
from src.preprocess import Preprocessor


class ComparativeAnalyzer:
    def __init__(self, ref_data_path):
        """
        Initialize the analyzer with reference data (SerenUplift) and global item stats

        Args:
            ref_data_path (str): Path to the reference data (SerenUplift)
        """
        self.ref_data = self._load_json(ref_data_path)
        _, self.post_data = Preprocessor().preprocess()
        self.item_counts = self.post_data['movieId'].value_counts()
        self.total_items = len(self.item_counts)
        self.popularity_scores = 1.0 / self.item_counts

    def _load_json(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def analyze(self, target_data_path, model_name):
        """
        Analyze a target model against the reference (filtering by Top-K).

        Args:
            target_data_path (str): Path to the baseline recommendation list
            model_name (str): Name of the baseline model

        Returns:
            dict: A Dictionary containing the analysis results
        """
        target_data = self._load_json(target_data_path)

        seren_scores = []
        unique_items = set()
        pop_sum = 0.0
        pop_count = 0

        common_users = 0

        for user_id, ref_items in self.ref_data.items():
            k = len(ref_items)
            target_items = target_data[user_id][:k]

            common_users += 1

            for item in target_items:
                seren_scores.append(float(item['serendipity_rating']))
                unique_items.add(item['title'])

                movie_id = item.get('org_movie_id')
                if movie_id in self.popularity_scores:
                    pop_sum += self.popularity_scores[movie_id]
                    pop_count += 1

        avg_seren = np.mean(seren_scores)
        coverage_ratio = len(unique_items) / self.total_items
        avg_pop = pop_sum / pop_count

        return {
            "Model": model_name,
            "Popularity": avg_pop,
            "Coverage": coverage_ratio,
            "Serendipity": avg_seren,
            "Unique_Items": len(unique_items),
            "Users": common_users
        }
