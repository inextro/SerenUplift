import os
import re
import time
import json
import yaml
import pandas as pd
from openai import OpenAI
from src.preprocess import Preprocessor


def load_id_map(map_path):
    df = pd.read_csv(map_path, sep=' ')
    return df.set_index('org_id')['remap_id'].to_dict()


class SerendipityEvaluator:
    def __init__(
        self, user_map_path, movie_map_path, movie_metadata_path, prompt_path,
        api_key=None, llm='gpt-5-nano'
    ):
        """
        description

        Args:
            user_map_path (str): Path to the user list(org_id remap_id) file
            movie_map_path (str): Path to the id map(org_id:remap_id) file
            movie_metadata_path (str): Path to the movie metadata file
            prompt_path (str): Path to the prompt file
            api_key (str, optional): OpenAI API key (default: None)
            llm (str, optional): llm name served by OpenAI (default: 'gpt-5-nano')
        """
        self.api_key = api_key
        self.llm = llm
        self.movie_metadata_path = movie_metadata_path
        self.user_remap_to_org = self._load_user_map(user_map_path)
        self.movie_remap_to_org = self._load_movie_map(movie_map_path)
        self.movie_metadata = self._load_movie_metadata(
            movie_map_path=movie_map_path, movie_metadata_path=movie_metadata_path
        )
        self.prompt = self._load_prompt(prompt_path)
        self.client = OpenAI(api_key=self.api_key)

    def _load_user_map(self, user_map_path):
        user_remap_to_org = {}
        with open(user_map_path, 'r') as f:
            next(f) # Skip header
            for line in f:
                parts = line.strip().split()
                org_id = int(parts[0])
                remap_id = int(parts[1])
                user_remap_to_org[remap_id] = org_id

        return user_remap_to_org

    def _load_movie_map(self, movie_map_path):
        movie_remap_to_org = {}
        with open(movie_map_path, 'r') as f:
            next(f) # Skip header
            for line in f:
                parts = line.strip().split()
                org_id = int(parts[0])
                remap_id = int(parts[1])
                movie_remap_to_org[remap_id] = org_id

        return movie_remap_to_org

    def _load_movie_metadata(self, movie_map_path, movie_metadata_path):
        """
        Load and filter movie metadata that exist in the id map

        Args:
            movie_map_path (str): Path to the id map file
            movie_metadata_path (str): Path to the movie metadata file

        Returns:
            pd.DataFrame: Filtered movie metadata with 'remap_id' column, filtered by valid org_ids

        """
        movie_map = load_id_map(movie_map_path)
        metadata = pd.read_csv(movie_metadata_path)

        valid_org_ids = set(movie_map.keys())
        filtered_metadata = metadata[metadata['movieId'].isin(valid_org_ids)].copy()
        filtered_metadata['remap_id'] = filtered_metadata['movieId'].map(movie_map)

        return filtered_metadata.set_index('remap_id')

    def _load_prompt(self, prompt_path):
        with open(prompt_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_user_history(self, user_history_path, history_limit=40):
        """
        Load user rating history automatically dispatching by file extension

        Args:
            user_history_path (str): Path to the user rating history file
            history_limit (int): Max number of history items to keep (default: 40)

        Returns:
            dict: User history dictionary {org_user_id: [(title, [genres]), ...]}

            Example) {
            0: [
                ('Toy Story', ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy']),
                ('Jumanji', ['Adventure', 'Children', 'Fantasy']), ...
                ],
            1: ...
            }
        """
        extension = os.path.splitext(user_history_path)[1]

        if extension == '.csv':
            return self._load_history_from_csv(
                history_path=user_history_path, history_limit=history_limit
            )

        elif extension == '.txt':
            return self._load_history_from_txt(
                history_path=user_history_path, history_limit=history_limit
            )

    def _load_history_from_csv(self, history_path, history_limit):
        user_history = {}
        meta_df = pd.read_csv(self.movie_metadata_path).set_index('movieId')

        history_df = pd.read_csv(history_path)
        preprocessor = Preprocessor()
        _, history_df = preprocessor.preprocess()

        if 'tstamp' in history_df.columns:
            history_df = history_df.sort_values('tstamp')

        for org_user_id, group in history_df.groupby('userId'):
            org_movie_ids = group['movieId'].tolist()

            if len(org_movie_ids) > history_limit:
                org_movie_ids = org_movie_ids[-history_limit:]

            movies = []
            for org_movie_id in org_movie_ids:
                title = meta_df.loc[org_movie_id, 'title']
                genres = meta_df.loc[org_movie_id, 'genres']
                try:
                    genres_list = genres.split('|')
                except AttributeError:
                    genres_list = ['(no genres listed)'] # if no genres are tagged

                movies.append((title, genres_list))

            user_history[org_user_id] = movies

        return user_history

    def _load_history_from_txt(self, history_path, history_limit):
        user_history = {}

        with open(history_path, 'r') as f:
            for line in f:
                parts = list(map(int, line.strip().split()))

                remap_user_id = parts[0]
                remap_movie_ids = parts[1:]

                org_user_id = self.user_remap_to_org[remap_user_id]

                if len(remap_movie_ids) > history_limit:
                    remap_movie_ids = remap_movie_ids[-history_limit:]

                movies = []
                for remap_movie_id in remap_movie_ids:
                    title = self.movie_metadata.loc[remap_movie_id, 'title']
                    genres = self.movie_metadata.loc[remap_movie_id, 'genres']
                    genres_list = genres.split('|')

                    movies.append((title, genres_list))

                user_history[org_user_id] = movies

        return user_history

    def _construct_messages(
        self, history_movies, candidate_title, candidate_genres, history_length=40
    ):
        """
        Construct messages for the LLM input

        Args:
            history_movies (list): List of tuples (title, genres) for user history
            candidate_title (str): Title of the candidate movie
            candidate_genres (list): List of genres of the candidate movie

        Returns:
            list: List of message dictionaries formatted for the OpenAI API
                Each dictionary contains:
                    - 'role' (str): The role of the message sender ('system' or 'user')
                    - 'content' (str): The content of the message, including the system instruction
                        or the user prompt with history and candidate movie details
        """
        prompts = self.prompt['serendipity_eval']
        system_template = prompts['system']
        user_template = prompts['user']

        # History format: "('Title', ['Genre1', 'Genre2'])"
        history_list = []
        for title, genres in history_movies[-history_length:]:
            genres_str = str(genres) # ['Action', 'Comedy'] -> "['Action', 'Comedy']"
            history_list.append(f'("{title}", {genres_str})')

        history_formatted = "[" + ", ".join(history_list) + "]"

        # Candidate format: "('Title', ['Genre1', 'Genre2'])"
        candidate_genres_str = str(candidate_genres)
        candidate_formatted = f'("{candidate_title}", {candidate_genres_str})'

        user_content = user_template.format(
            **{
                'user_rated_movies_list': history_formatted,
                'recommended_movie_info': candidate_formatted
            }
        )

        messages = [
            {'role': 'system', 'content': system_template},
            {'role': 'user', 'content': user_content}
        ]

        return messages

    def evaluate(
        self, recommendations, user_history_path, output_path=None, dry_run=False
    ):
        """
        Evaluate recommendations

        Args:
            recommendations (dict): Recommendation results {remap_user_id (str): [remap_movie_id (int), ...]}
            user_history_path (str): Path to user history csv
            output_path (str, optional): Path to save results
            dry_run (bool): If True, do not call API

        Returns:
            dict: Evaluation results {remap_user_id: [{'remap_movie_id': int, 'title': str, 'serendipity_rating': int}]}

        """
        print(f"Loading user history from {user_history_path}...")
        user_history_map = self._load_user_history(user_history_path)
        print("User history loaded.")

        results = {}
        total_users = len(recommendations)

        processed_count = 0
        for i, (remap_user_id, rec_items) in enumerate(recommendations.items()):
            remap_user_id = int(remap_user_id)
            org_user_id = self.user_remap_to_org[remap_user_id]

            history_movies = user_history_map[org_user_id]
            user_results = []

            if not isinstance(rec_items, list):
                rec_items = [rec_items]

            print(f"[{i + 1}/{total_users}] Processing User (Org_id: {org_user_id}, Remap_id: {remap_user_id})...")

            for remap_movie_id in rec_items:
                org_movie_id = self.movie_remap_to_org.get(remap_movie_id)

                item_info = self.movie_metadata.loc[remap_movie_id]
                candidate_title = item_info['title']
                candidate_genres = item_info['genres']
                candidate_genres = candidate_genres.split('|')

                messages = self._construct_messages(
                    history_movies=history_movies,
                    candidate_title=candidate_title,
                    candidate_genres=candidate_genres
                )

                serendipity_rating = None

                if dry_run:
                    serendipity_rating = 3 # Dummy rating for dry run
                    if processed_count < 3:
                        print(f'\n--- Dry Run Prompt [User OrgID: {org_user_id} | Movie OrgID: {org_movie_id}] ---')
                        print(json.dumps(messages, indent=2, ensure_ascii=False))
                        time.sleep(0.01)
                else:
                    response = self.client.chat.completions.create(
                        model=self.llm, messages=messages, temperature=0, max_tokens=10
                    )
                    content = response.choices[0].message.content
                    if content:
                        match = re.search(r'\b[1-5]\b', content)
                        if match:
                            serendipity_rating = int(match.group())
                    time.sleep(0.2)

                user_results.append({
                    'org_movie_id': int(org_movie_id),
                    'title': candidate_title,
                    'genres': candidate_genres,
                    'serendipity_rating': serendipity_rating
                })

            results[org_user_id] = user_results
            processed_count += 1

            if dry_run and processed_count >= 3:
                break

        total_serendipity_score = 0
        count = 0
        for result in results.values():
            for item in result:
                if item['serendipity_rating'] is not None:
                    total_serendipity_score += item['serendipity_rating']
                    count += 1
        print(f'Average Serendipity Score: {total_serendipity_score / count:.4f}')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        return results
