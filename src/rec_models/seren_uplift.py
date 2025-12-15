import pandas as pd
from tqdm import tqdm
from itertools import product
from ..preprocess import Preprocessor
from .abstract_recommender import AbstractRecommender


class SerenUplift(AbstractRecommender):
    def __init__(self, df='./data/processed_data/pred_uplift.csv'):
        """
        Initialize the recommender with the preprocessed dataframe
        It should already be filtered for unseen items

        Args:
            df (pd.DataFrame): A dataframe containing columns ['remap_userId', 'remap_movieId', 'uplift_score', 'pred_post', 'pred_prior']
        """
        self.uplift = self._preprocess(df)
        print(f'Recommender initialized with {self.uplift.shape[0]:,} pairs')

    @classmethod
    def load_from_path(
        cls, uplift_path='./data/processed_data/pred_uplift.csv'
    ):
        """
        Returns a SerenUplift instance loaded from the given path

        Args:
            uplift_path (str): Path to the uplift result file

        Returns:
            SerenUplift: A SerenUplift instance initialized with the given uplift result file
        """
        uplift = cls._preprocess(uplift_path)
        return cls(uplift)

    def _preprocess(self, uplift_path='./data/processed_data/pred_uplift.csv'):
        """
        Preprocess the uplift data by filtering out unseen items

        Args:
            uplift_path (str): Path to the uplift result file

        Returns:
            pd.DataFrame: A preprocessed dataframe containing columns ['remap_userId', 'remap_movieId', 'uplift_score', 'pred_post', 'pred_prior']
        """
        uplift = pd.read_csv(uplift_path)
        print(f'Total number of combinations: {uplift.shape[0]:,}')

        preprocessor = Preprocessor()
        _, post_data = preprocessor.preprocess()
        user_map, movie_map = preprocessor.id_mapping(df=post_data)

        post_data['remap_userId'] = post_data['userId'].map(user_map)
        post_data['remap_movieId'] = post_data['movieId'].map(movie_map)

        print(f'Total number of combination (before filtering): {uplift.shape[0]:,}')
        print(f'Total number of post ratings: {post_data.shape[0]:,}')

        filtered_uplift = pd.merge(
            left=uplift, right=post_data,
            on=['remap_userId', 'remap_movieId'],
            how='left', indicator=True
        )
        filtered_uplift = filtered_uplift[filtered_uplift['_merge'] == 'left_only']
        filtered_uplift = filtered_uplift.iloc[:, :5].reset_index(drop=True)

        print(f'Total number of combinations (after filtering): {filtered_uplift.shape[0]:,}')

        return filtered_uplift

    def get_candidate_stats(
        self,
        uplift_thresholds=[2.0, 2.5, 3.0],
        post_thresholds=[4.0, 4.5]
    ):
        """
        Filter recommendation candidates based on the given thresholds
        Prints statistics for each threshold combinations

        Args:
            uplift_thresholds (list[float]): Threshold values for uplift score (Default: 2.0)
            post_thresholds (list[float]): Threshold values for predicted post-rating (Default: 4.0)
        """
        print(f'| {"Uplift Threshold":^16} | {"Post Threshold":^14} | {"#Candidates":^12} | {"#Users":^6} | {"Min":^6} | {"Max":^6} | {"Mean":^6} | {"Median":^6} |')

        combinations = list(product(uplift_thresholds, post_thresholds))
        for uplift_threshold, post_threshold in tqdm(combinations):
            serendipity_condition = (
                (self.uplift['uplift_score'] >= uplift_threshold) &
                (self.uplift['pred_post'] >= post_threshold)
            )

            candidates = self.uplift[serendipity_condition]
            user_counts = candidates['remap_userId'].value_counts()

            tqdm.write(
                f'| {uplift_threshold:>16.1f} | {post_threshold:>14.1f} | '
                f'{candidates.shape[0]:>12,} | {len(user_counts):>6,} | '
                f'{user_counts.min():>6,} | {user_counts.max():>6,} | '
                f'{user_counts.mean():>6,.2f} | {user_counts.median():>6} |'
            )

    def recommend(
        self, top_k=20, uplift_threshold=2.0, post_threshold=4.0, **kwargs
    ):
        """
        Recommend top-k items for each user

        If fewer than k candidates found, return all available candidates
        Otherwise, returns the top-k candidates sorted by uplift score descending

        Args:
            k (int): Number of items to recommend for each user (default: 20)
            uplift_threshold (float): Threshold value for uplift score (default: 2.0)
            post_threshold (float): Threshold value for predicted post-rating (default: 4.0)

        Returns:
            dict: A dictionary mapping user IDs to a list of recommended item IDs
                Example: {user_id: [item_id_1, ..., item_id_k]}
        """
        serendipity_condition = (
            (self.uplift['uplift_score'] >= uplift_threshold) &
            (self.uplift['pred_post'] >= post_threshold)
        )

        candidates = self.uplift[serendipity_condition].copy()
        candidates = candidates.sort_values(
            by=['remap_userId', 'uplift_score'],
            ascending=[True, False]
        )

        top_k_candidates = candidates.groupby('remap_userId').head(top_k)
        recommendations = (
            top_k_candidates.groupby('remap_userId')['remap_movieId']
            .apply(list).to_dict()
        )

        return recommendations
