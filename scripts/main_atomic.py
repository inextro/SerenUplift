import os
import argparse
import pandas as pd
from src.utils import load_id_map
from src.preprocess import Preprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--user_map_path', type=str, default='lgcn/data/movielens/user_list.txt',
        help='Path to user map file'
    )
    parser.add_argument(
        '--movie_map_path', type=str, default='lgcn/data/movielens/item_list.txt',
        help='Path to movie map file'
    )
    parser.add_argument(
        '--save_dir', type=str, default='data/processed_data/',
        help='Path to save atomic data'
    )
    args, _ = parser.parse_known_args()

    return args


def main(user_map_path, movie_map_path, save_dir):
    user_map = load_id_map(user_map_path)
    movie_map = load_id_map(movie_map_path)

    preprocessor = Preprocessor()
    _, core_ratings = preprocessor.preprocess()

    core_ratings['user_id:token'] = core_ratings['userId'].map(user_map)
    core_ratings['item_id:token'] = core_ratings['movieId'].map(movie_map)

    core_ratings['user_id:token'] = core_ratings['user_id:token'].astype(int)
    core_ratings['item_id:token'] = core_ratings['item_id:token'].astype(int)
    core_ratings['rating:float'] = core_ratings['rating'].astype(float)
    core_ratings['timestamp:float'] = pd.to_datetime(core_ratings['tstamp']).astype(int) // 10**9

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'ml-belief.inter')

    df = core_ratings[['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']]
    df.to_csv(save_path, sep='\t', index=False)


if __name__ == '__main__':
    args = parse_args()
    main(
        user_map_path=args.user_map_path,
        movie_map_path=args.movie_map_path,
        save_dir=args.save_dir
    )
