from xgboost import XGBRegressor
from preprocess import Preprocessor
from t_learner import TLearner


def main(configs, user_emb_path, movie_emb_path):
    preprocessor = Preprocessor()
    prior_data, post_data = preprocessor.preprocess()

    prior_model = XGBRegressor(**configs)
    post_model = XGBRegressor(**configs)
    
    t_learner = TLearner(
        prior_model=prior_model, post_model=post_model, 
        user_emb_path=user_emb_path, movie_emb_path=movie_emb_path
    )

    t_learner.fit(
        prior_data=prior_data, post_data=post_data
    )

    df_uplift = t_learner.compute_uplift()
    df_uplift.to_csv('./data/processed_data/pred_uplift.csv', index=False)

if __name__ == '__main__':
    configs = {
        'n_estimators': 5000, 
        'early_stopping_rounds': 50
    }

    print('='*100)
    print(f'Current Configurations: ')
    for key, value in configs.items():
        print(f'{key}: {value}')
    print('='*100)

    main(
        configs=configs, 
        user_emb_path='./embeddings/user_emb_16.npy', 
        movie_emb_path='./embeddings/movie_emb_16.npy'
    )