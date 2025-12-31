import json
import numpy as np
from scipy import stats
from src.preprocess import Preprocessor


class ComparativeAnalyzer:
    def __init__(self, ref_data_path):
        """
        Initialize the analyzer with reference/target data (SerenUplift) and global item stats

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

    def _get_paired_scores(self, target_data):
        """
        Helper method to retrieve aligned user-level mean serendipity scores
        Target items are adaptively filtered by top-k based on reference items count

        Returns:
            tuple: (ref_user_scores, target_user_scores)
        """
        ref_user_scores = []
        target_user_scores = []

        for user_id, ref_items in self.ref_data.items():
            u_ref = [
                float(item['serendipity_rating'])
                for item in ref_items
            ]

            k = len(ref_items) # adaptive k
            target_items = target_data[user_id][:k]
            u_target = [
                float(item['serendipity_rating'])
                for item in target_items
            ]

            ref_user_scores.append(np.mean(u_ref))
            target_user_scores.append(np.mean(u_target))

        return ref_user_scores, target_user_scores

    def analyze(self, model_name, target_data_path):
        """
        Analyze a target model against the reference (filtering by Top-K).

        Args:
            model_name (str): Name of the baseline model
            target_data_path (str): Path to the target recommendation list (baseline)

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

    def paired_t_test(self, target_data_path):
        """
        Perform a paired t-test between the reference and target recommendation list

        Args:
            target_data_path (str): Path to the target recommendation list (baseline)

        Returns:
            dict: A dictionary containing the t-test results
        """
        target_data = self._load_json(target_data_path)
        ref_user_scores, target_user_scores = self._get_paired_scores(target_data)

        t_stat, p_value = stats.ttest_rel(ref_user_scores, target_user_scores)

        return {
            't_stat': t_stat,
            'p_value': p_value
        }

    def cohen_d(self, target_data_path):
        """
        Calculate Cohen's d between the reference and target recommendation list

        Args:
            target_data_path (str): Path to the target recommendation list (baseline)

        Returns:
            dict: A dictionary containing the Cohen's d results
        """
        target_data = self._load_json(target_data_path)
        ref_user_scores, target_user_scores = self._get_paired_scores(target_data)

        n1 = len(ref_user_scores)
        n2 = len(target_user_scores)

        ref_std = np.std(ref_user_scores, ddof=1)
        target_std = np.std(target_user_scores, ddof=1)

        pooled_std = np.sqrt(
            ((n1 - 1) * ref_std**2 + (n2 - 1) * target_std**2) / (n1 + n2 - 2)
        )

        diff = np.array(ref_user_scores) - np.array(target_user_scores)
        mean_diff = np.mean(diff)

        d = mean_diff / pooled_std

        return {'cohen_d': d}

    def wilcoxon_test(self, target_data_path):
        """
        Perform a Wilcoxon signed-rank test between the reference and target recommendation list.
        Robust to non-normality (e.g., Likert scale data).

        Args:
            target_data_path (str): Path to the target recommendation list (baseline)

        Returns:
            dict: A dictionary containing the Wilcoxon test results
        """
        target_data = self._load_json(target_data_path)
        ref_user_scores, target_user_scores = self._get_paired_scores(target_data)

        stat, p_value = stats.wilcoxon(ref_user_scores, target_user_scores)

        return {
            'stat': stat,
            'p_value': p_value
        }
