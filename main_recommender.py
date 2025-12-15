import pandas as pd
from src.preprocess import Preprocessor
from src.recommender import SerenRecommender


def main(uplift_thresholds, post_thresholds):
    df = _preprocess()
    recommender = SerenRecommender(df=df)
    recommender.evaluate(
        uplift_thresholds=uplift_thresholds, post_thresholds=post_thresholds
    )

def _preprocess(uplift_path='./data/processed_data/pred_uplift.csv'):
    """
    
    Returns:
        pd.DataFrame: 
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


if __name__ == '__main__':
    main(
        uplift_thresholds=[2.0, 2.5, 3.0], 
        post_thresholds=[4.0, 4.5]
    )