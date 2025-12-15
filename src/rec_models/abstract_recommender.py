from abc import ABC, abstractmethod


class AbstractRecommender(ABC):
    """
    Abstract base class for recommender models with pre-trained weights
    """
    def __init__(self, config):
        """
        Initialize the recommender with the given configuration

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.model = self._load_model()

    @abstractmethod
    def recommend(
        self, top_k=20, **kwargs
    ):
        """
        Recommend top-k items for each user.

        Args:
            k (int): Number of items to recommend for each user (default: 20)
            **kwargs: Additional keyword arguments

        Returns:
            dict: A dictionary mapping user IDs to a list of recommended item IDs
                Example: {user_id: [item_id_1, ..., item_id_k]}
        """
        pass
