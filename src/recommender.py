from tqdm import tqdm
from itertools import product


class SerenRecommender:
    def __init__(self, df):
        """
        Initialize the recommender with the preprocessed dataframe.
        It should already be filtered for unseen items.

        Args: 
            df (pd.DataFrame): A dataframe containing columns ['remap_userId', 'remap_movieId', 'uplift_score', 'pred_post', 'pred_prior']
        """
        self.uplift = df.copy()
        print(f'Recommender initialized with {self.uplift.shape[0]:,} pairs')

    def evaluate(
        self, uplift_thresholds=[2.0], post_thresholds=[4.0], save_plots=False
    ):
        """
        Filter recommendation candidates based on the given thresholds.
        Prints statistics for each threshold combinations
        
        Args: 
            uplift_thresholds (list[float]): Threshold values for uplift score (Default: 2.0)
            post_thresholds (list[float]): Threshold values for predicted post-rating (Default: 4.0)
            save_plots (bool): Flag for saving plots
        """
        print(f'| {'Uplift Threshold':^16} | {'Post Threshold':^14} | {'#Candidates':^12} | {'#Users':^6} | {'Min':^6} | {'Max':^6} | {'Mean':^6} | {'Median':^6} |')
        
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