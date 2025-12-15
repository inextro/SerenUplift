import numpy as np
from .abstract_recommender import AbstractRecommender


class LightGCN(AbstractRecommender):
    def __init__(self, embedding_dims=16):
        user_emb_path = f'embeddings/user_emb_{embedding_dims}.npy'
        movie_emb_path = f'embeddings/movie_emb_{embedding_dims}.npy'

        self.user_embs = np.load(user_emb_path)
        self.item_embs = np.load(movie_emb_path)
        self.history = self._load_history()

    def _load_history(self, history_path='lgcn/data/movielens/train.txt'):
        history = {}
        with open(history_path, 'r') as f:
            for line in f:
                histories = list(map(int, line.strip().split()))

                user_id = histories[0]
                movie_ids = histories[1:]

                history[user_id] = set(movie_ids)

        return history

    def recommend(self, top_k=20, **kwargs):
        """
        Recommend top-k items for each user

        Args:
            top_k (int): Number of items to recommend for each user (default: 20)

        Returns:
            dict: A dictionary mapping user IDs to a list of recommended item IDs
                Example: {user_id: [item_id_1, ..., item_id_k]}
        """
        recommendations = {}

        num_users = self.user_embs.shape[0]
        for user_id in range(num_users):
            user_emb = self.user_embs[user_id]

            scores = np.dot(
                self.item_embs, user_emb
            )

            watched_items = list(self.history[user_id])
            scores[watched_items] = -np.inf

            top_k_candidates = np.argpartition(scores, -top_k)[-top_k:]

            top_k_scores = scores[top_k_candidates]
            sorted_indices = np.argsort(top_k_scores)[::-1]

            top_k_candidates = top_k_candidates[sorted_indices]
            recommendations[user_id] = top_k_candidates.tolist()

        return recommendations
