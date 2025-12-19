import os
import re
import json
import yaml
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from src.preprocess import Preprocessor


class SerendipityEvaluator:
    def __init__(
        self, user_map_path, movie_map_path, movie_metadata_path, prompt_path, api_key, llm
    ):
        """
        Initialize the SerendipityEvaluator with mapping files, metadata, and LLM configuration

        Args:
            user_map_path (str): Path to the user list(org_id remap_id) file
            movie_map_path (str): Path to the id map(org_id:remap_id) file
            movie_metadata_path (str): Path to the movie metadata file
            prompt_path (str): Path to the prompt file
            api_key (str): OpenAI API key
            llm (str): llm name served by OpenAI
        """
        self.api_key = api_key
        self.llm_client = AsyncOpenAI(api_key=self.api_key)
        self.llm_model = llm
        self.movie_metadata_path = movie_metadata_path
        self.user_remap_to_org = self._load_user_map(user_map_path)
        self.movie_remap_to_org = self._load_movie_map(movie_map_path)
        self.movie_metadata = self._load_movie_metadata(
            movie_map_path=movie_map_path, movie_metadata_path=movie_metadata_path
        )
        self.prompt = self._load_prompt(prompt_path)

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
        metadata = pd.read_csv(movie_metadata_path)

        valid_org_ids = set(self.movie_remap_to_org.values())
        filtered_metadata = metadata[metadata['movieId'].isin(valid_org_ids)].copy()

        return filtered_metadata.set_index('movieId')

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
                    org_movie_id = self.movie_remap_to_org[remap_movie_id]

                    title = self.movie_metadata.loc[org_movie_id, 'title']
                    genres = self.movie_metadata.loc[org_movie_id, 'genres']
                    try:
                        genres_list = genres.split('|')
                    except AttributeError:
                        genres_list = ['(no genres listed)'] # if no genres are tagged

                    movies.append((title, genres_list))

                user_history[org_user_id] = movies

        return user_history

    def _construct_messages(
        self, history_movies, candidate_title, candidate_genres, history_length=40
    ):
        """
        Construct messages for the LLM input

        Args:
            history_movies (list[tuple[str, list[str]]]): List of tuples (title, genres) for user history
            candidate_title (str): Title of the candidate movie
            candidate_genres (list[str]): List of genres of the candidate movie

        Returns:
            list[dict[str, str]]: List of message dictionaries formatted for the OpenAI API
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

    async def _evaluate_single_recommendation(
        self, messages, org_user_id, org_movie_id,
        candidate_title, candidate_genres, semaphore, dry_run
    ):
        """
        Evaluate a single recommendation asynchronously

        Args:
            messages (list[dict]): A list of dictionaries containing the system instructions and the evaluation query
            org_user_id (int): Original user ID
            org_movie_id (int): Original movie ID
            candidate_title (str): The title of the candidate movie
            candidate_genres (list[str]): A list of genre strings associated with the candidate movie
            semaphore (asyncio.Semaphore): A semaphore to control the number of concurrent API calls
            dry_run (bool): If True, returns dummy result; not calling API

        Returns:
            dict: Evaluation result including serendipity rating
        """
        async with semaphore:
            if dry_run:
                return {
                    'org_user_id': org_user_id,
                    'org_movie_id': int(org_movie_id),
                    'title': candidate_title,
                    'genres': candidate_genres,
                    'serendipity_rating': 3
                }

            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0,
                max_tokens=10
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r'\b([1-5])\b', content)
            serendipity_rating = int(match.group(1))

            return {
                'org_user_id': org_user_id,
                'org_movie_id': int(org_movie_id),
                'title': candidate_title,
                'genres': candidate_genres,
                'serendipity_rating': serendipity_rating
            }

    async def evaluate(
        self, recommendations, user_history_path, output_path=None, dry_run=False, concurrency=50
    ):
        """
        Evaluate recommendations with async concurrency

        Args:
            recommendations (dict[str, list[int]]): Recommendation results {remap_user_id (str): [remap_movie_id (int), ...]}
            user_history_path (str): Path to user history csv
            output_path (str): Path to save results
            dry_run (bool): If True, do not call API
            concurrency (int): Max number of concurrent API requests

        Returns:
            dict[int, list[dict]]: Evaluation results {remap_user_id: [{'remap_movie_id': int, 'title': str, 'serendipity_rating': int}]}

        """
        print(f"Loading user history from {user_history_path}...")
        user_history_map = self._load_user_history(user_history_path)
        print("User history loaded.")

        tasks = []
        results = {}
        semaphore = asyncio.Semaphore(concurrency)

        for remap_user_id, rec_items in recommendations.items():
            remap_user_id = int(remap_user_id)
            org_user_id = self.user_remap_to_org[remap_user_id]

            history_movies = user_history_map[org_user_id]

            if not isinstance(rec_items, list):
                rec_items = [rec_items]

            for remap_movie_id in rec_items:
                org_movie_id = self.movie_remap_to_org.get(remap_movie_id)

                item_info = self.movie_metadata.loc[org_movie_id]
                candidate_title = item_info['title']
                candidate_genres = item_info['genres']
                candidate_genres = candidate_genres.split('|')

                messages = self._construct_messages(
                    history_movies=history_movies,
                    candidate_title=candidate_title,
                    candidate_genres=candidate_genres
                )

                if dry_run and len(tasks) < 3:
                    tqdm.write(f'\n--- Dry Run Prompt [Org_UserID: {org_user_id}, Org_MovieID: {org_movie_id}] ---')
                    tqdm.write(json.dumps(messages, indent=2, ensure_ascii=False))

                tasks.append(
                    self._evaluate_single_recommendation(
                        messages=messages, org_user_id=org_user_id, org_movie_id=org_movie_id,
                        candidate_title=candidate_title, candidate_genres=candidate_genres,
                        semaphore=semaphore, dry_run=dry_run
                    )
                )

        if dry_run:
            print(f'[Dry Run] Prepared {len(tasks)} tasks')
        else:
            print(f'Starting execution of {len(tasks)} tasks (concurrency={concurrency})')

        raw_results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            raw_result = await f
            if raw_result:
                raw_results.append(raw_result)

        results = {}
        for raw_result in raw_results:
            org_user_id = raw_result['org_user_id']

            if org_user_id not in results:
                results[org_user_id] = []

            clean_result = {k: v for k, v in raw_result.items() if k != 'org_user_id'}
            results[org_user_id].append(clean_result)

        count = 0
        total_score = 0
        for user_items in results.values():
            for item in user_items:
                if item['serendipity_rating'] is not None:
                    total_score += item['serendipity_rating']
                    count += 1

        print(f'Average serendipity score: {total_score / count:.4f}')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        return results
