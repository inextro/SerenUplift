import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from .utils import load_id_map


class TLearner():
    """
    A class to predict uplift scores using T-learner

    uplift_score = pred_post - pred_prior
    """
    def __init__(self, prior_model, post_model, user_emb_path, movie_emb_path):
        """
        Args:
            prior_model (object): Any regression model object to predict prior rating
            post_model (object): Any regression model object to predict post rating
            user_emb_path (str): User embedding path (.npy)
            movie_emb_path (str): Movie embedding path (.npy)
        """
        self.prior_model = prior_model
        self.post_model = post_model
        self.user_embs = np.load(user_emb_path)
        self.movie_embs = np.load(movie_emb_path)
        self.user_map = load_id_map('./lgcn/data/movielens/user_list.txt')
        self.movie_map = load_id_map('./lgcn/data/movielens/item_list.txt')

    def fit(self, prior_data, post_data):
        """
        Fit prior model and post model

        Args:
            prior_data (pd.DataFrame): (Preprocessed) Prior rating data
            post_data (pd.DataFrame): (Preprocessed) Post rating data
        """
        print('Start fitting prior model')
        X_prior = self._create_feature(prior_data)
        y_prior = prior_data['userPredictRating'].values

        X_train_prior, X_test_prior, y_train_prior, y_test_prior = train_test_split(
            X_prior, y_prior, test_size=0.2, random_state=42
        )
        self.prior_model.fit(X_train_prior, y_train_prior, eval_set=[(X_test_prior, y_test_prior)])

        print('Start fitting post model')
        X_post = self._create_feature(post_data)
        y_post = post_data['rating'].values

        X_train_post, X_test_post, y_train_post, y_test_post = train_test_split(
            X_post, y_post, test_size=0.2, random_state=42
        )
        self.post_model.fit(X_train_post, y_train_post, eval_set=[(X_test_post, y_test_post)])

        mse_prior = mean_squared_error(
            y_true=y_test_prior, y_pred=self.prior_model.predict(X_test_prior)
        )
        mse_post = mean_squared_error(
            y_true=y_test_post, y_pred=self.post_model.predict(X_test_post)
        )

        print(f'MSE of prior model: {mse_prior:.4f}')
        print(f'MSE of post model: {mse_post:.4f}')
        print(f'RMSE of prior model: {np.sqrt(mse_prior):.4f}')
        print(f'RMSE of post model: {np.sqrt(mse_post):.4f}')

    def compute_uplift(self, n_jobs=6):
        """
        Compute uplift score for all possible combinations of (user X movie)

        Args:
            n_jobs (int): A number of CPU cores to be used

        Returns:
            pd.DataFrame: A dataframe with columns of ['remap_userId', 'remap_movieId', 'uplift_score', 'pred_prior', 'pred_post']
        """
        num_users = self.user_embs.shape[0]
        num_movies = self.movie_embs.shape[0]

        print(f'Compute uplift scores for {num_users} users X {num_movies} movies')
        print(f'Total number of combinations: {num_users * num_movies:,}')

        n_chunks = n_jobs * 4
        user_chunks = np.array_split(range(num_users), n_chunks)

        uplift_scores = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(self._process_batch)(user_chunk)
            for user_chunk in tqdm(user_chunks, desc='Parallel Processing')
        )
        flat_uplift_scores = [score for uplift_score in uplift_scores for score in uplift_score]

        df = pd.DataFrame(
            flat_uplift_scores,
            columns=['remap_userId', 'remap_movieId', 'uplift_score', 'pred_prior', 'pred_post']
        )

        return df

    def _create_feature(self, df):
        """
        Create input feature for regression models to predict prior/post rating
        The input feature is created by concatenate [user_embedding, movie_embedding]

        Args:
            df (pd.DataFrame): A dataframe of prior/post rating

        Returns:

        """
        df = df.copy()

        df['remap_userId'] = df['userId'].map(self.user_map)
        df['remap_movieId'] = df['movieId'].map(self.movie_map)

        user_indices = df['remap_userId'].astype(int).values
        movie_indices = df['remap_movieId'].astype(int).values

        X_user = self.user_embs[user_indices]
        X_movie = self.movie_embs[movie_indices]

        X = np.concatenate((X_user, X_movie), axis=1)

        return X

    def _process_batch(self, user_indices):
        """
        Compute uplift scores for all movies for the assigned users

        Args:
            user_indices (np.ndarray): An array of user indices to process

        Returns:
            list: A list of [user_idx, movie_idx, uplift_score]
        """
        batch_results = []
        num_movies = self.movie_embs.shape[0]

        for user_idx in user_indices:
            user_emb = np.tile(self.user_embs[user_idx], (num_movies, 1))

            X_batch = np.concatenate((user_emb, self.movie_embs), axis=1)

            pred_priors = self.prior_model.predict(X_batch)
            pred_posts = self.post_model.predict(X_batch)

            uplift_scores = pred_posts - pred_priors

            batch_results.extend([
                [user_idx, movie_idx, uplift_score, pred_prior, pred_post]
                for movie_idx, (uplift_score, pred_prior, pred_post) in enumerate(zip(uplift_scores, pred_priors, pred_posts))
            ])

        return batch_results
