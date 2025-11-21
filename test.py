import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from preprocess import Preprocessor



def load_id_map(id_map_path):
    return pd.read_csv(id_map_path, sep=' ').set_index('org_id')['remap_id'].to_dict()


class TLearner:
    def __init__(self, prior_model, post_model, user_emb_path, movie_emb_path):
        self.prior_model = prior_model
        self.post_model = post_model
        self.user_embs = np.load(user_emb_path)
        self.movie_embs = np.load(movie_emb_path)

    def _create_feature(self, df):
        user_indices = df['remap_userId'].values
        movie_indices = df['remap_movieId'].values

        X_user = self.user_embs[user_indices]
        X_movie = self.movie_embs[movie_indices]

        X = np.concatenate((X_user, X_movie), axis=1)

        return X

    def fit(self, prior_data, post_data):
        print('Start fitting prior model')
        X_prior = self._create_feature(prior_data)
        y_prior = prior_data['userPredictRating'].values

        self.prior_model.fit(X_prior, y_prior)

        print('Start fitting post model')
        X_post = self._create_feature(post_data)
        y_post = post_data['rating'].values       

        self.post_model.fit(X_post, y_post)

    def predict_uplift(self):
        pass


if __name__ == '__main__':
    preprocessor = Preprocessor()
    prior, post = preprocessor.preprocess()

    prior_model = XGBRegressor()
    post_model = XGBRegressor()

    t_learner = TLearner(
        prior_model=prior_model, 
        post_model=post_model, 
        user_meb_path='./embeddings/user_emb_16.npy', 
        movie_emb_path='./embeddings/movie_emb_16.npy'
    )
    
    user_map = load_id_map('./lgcn/data/movielens/user_lists.txt')
    movie_map = load_id_map('./lgcn/data/movielens/movie_lists.txt')

    prior['remap_userId'] = prior['userId'].map(user_map)
    prior['remap_movieId'] = prior['movieId'].map(movie_map)
    
    post['remap_userId'] = post['userId'].map(user_map)
    post['remap_movieId'] = post['movieId'].map(movie_map)

    prior_train, prior_test = train_test_split(prior, test_size=0.2, random_state=42)
    post_train, post_test = train_test_split(post, test_size=0.2, random_state=42)
    
    t_learner.fit(
        prior_data=prior_train, 
        post_data=post_train
    )