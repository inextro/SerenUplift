import os
import numpy as np
import pandas as pd
from tqdm import tqdm


class Preprocessor:
    def __init__(
        self, 
        prior_path='./data/raw_data/belief_data.csv', 
        post_path='./data/raw_data/user_rating_history.csv'
    ):
        self.prior = pd.read_csv(prior_path)
        self.post = pd.read_csv(post_path)


    def _k_core_filter(self, df, k):
        df = df.copy()

        while True:
            initial_size = df.shape[0]

            movie_counts = df['movieId'].value_counts()
            valid_movies = movie_counts[movie_counts >= k].index
            df = df[df['movieId'].isin(valid_movies)]

            user_counts = df['userId'].value_counts()
            valid_users = user_counts[user_counts >= k].index
            df = df[df['userId'].isin(valid_users)]

            if df.shape[0] == initial_size:
                break

        return df

    def id_mapping(self, df, output_path):
        unique_users = df['userId'].unique()
        user_map = {org_id: i for i, org_id in enumerate(unique_users)}

        unique_movies = df['movieId'].unique()
        movie_map = {org_id: i for i, org_id in enumerate(unique_movies)}

        user_list = pd.DataFrame(user_map.items(), columns=['org_id', 'remap_id'])
        user_list.to_csv(os.path.join(output_path, 'user_lists.txt'), sep=' ', index=False)

        movie_list = pd.DataFrame(movie_map.items(), columns=['org_id', 'remap_id'])
        movie_list.to_csv(os.path.join(output_path, 'item_list.txt'), sep=' ', index=False)

        return user_map, movie_map

    def preprocess(self):
        self.prior = self.prior[self.prior['isSeen'] == 0]
        self.prior = self.prior.drop_duplicates(subset=['userId', 'movieId'], keep='last', ignore_index=True)
        init_prior_date = self.prior['tstamp'].sort_values().iloc[0]

        self.post = self.post[self.post['rating'] != -1]
        self.post = self.post[~self.post['rating'].isna()]
        self.post = self.post.drop_duplicates(subset=['userId', 'movieId'], keep='last', ignore_index=True)
        self.post = self.post[self.post['tstamp'] < init_prior_date]

        core_ratings = self._k_core_filter(df=self.post, k=5)

        print('='*200)
        print(f'Total number of ratings(post): {core_ratings.shape[0]:,}')
        print(f'Total number of unique users(post): {core_ratings['userId'].nunique():,}')
        print(f'Total number of unique movies(post): {core_ratings['movieId'].nunique():,}')
        print('='*200)

        valid_users = set(core_ratings['userId'].unique())
        valid_movies = set(core_ratings['movieId'].unique())
        
        prior_filtered = self.prior[
            self.prior['userId'].isin(valid_users) & self.prior['movieId'].isin(valid_movies)
        ]

        print('='*200)
        print(f'Total number of ratings(prior): {prior_filtered.shape[0]:,}')
        print(f'Total number of unique users(prior): {prior_filtered['userId'].nunique():,}')
        print(f'Total number of unique movies(prior): {prior_filtered['movieId'].nunique():,}')

        return prior_filtered, core_ratings # processed_prior, processed_post

    def build_lgcn_dataset(self, df, test_ratio=0.2, seed=42, output_path='./lgcn/data/movielens/'):
        np.random.seed(seed)

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        user_map, movie_map = self.id_mapping(df=df, output_path=output_path)
        
        df['remap_userId'] = df['userId'].map(user_map)
        df['remap_movieId'] = df['movieId'].map(movie_map)

        train_file_path = os.path.join(output_path, 'train.txt')
        test_file_path = os.path.join(output_path, 'test.txt')

        grouped = df.groupby('remap_userId')['remap_movieId'].apply(list)

        with open(train_file_path, 'w') as train, open(test_file_path, 'w') as test:
            for user_id, items in tqdm(grouped.items()):
                np.random.shuffle(items)
                split_point = int(len(items) * (1 - test_ratio))

                train_items = items[:split_point]
                test_items = items[split_point:]

                train_line = f'{user_id} {' '.join(map(str, train_items))}\n'
                train.write(train_line)

                test_line = f'{user_id} {' '.join(map(str, test_items))}\n'
                test.write(test_line)


if __name__ == '__main__':
    preprocessor = Preprocessor()
    prior_filtered, core_ratings = preprocessor.preprocess()
    preprocessor.build_lgcn_dataset(df=core_ratings)