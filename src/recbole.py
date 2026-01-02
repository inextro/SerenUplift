import os
import json
import torch
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer
from recbole.utils.case_study import full_sort_topk


class RecboleRunner:
    def __init__(self, config):
        self.config = config
        self.dataset = create_dataset(config)
        self.train_data, self.valid_data, self.test_data = data_preparation(
            config=config, dataset=self.dataset
        )
        self.model = get_model(self.config['model'])(config, self.train_data.dataset).to(self.config['device'])
        self.trainer = get_trainer(self.config['MODEL_TYPE'], self.config['model'])(self.config, self.model)

    def train(self):
        best_valid_score, best_valid_result = self.trainer.fit(
            self.train_data, self.valid_data, show_progress=True
        )
        print(f'Training completed. Best valid results: {best_valid_result}')

        return best_valid_score, best_valid_result

    def inference(self, top_k, device, save_path):
        self.model.eval()

        print(f'Generating Top-{top_k} recommendations...')

        recommendations = {}

        batch_size = self.config['eval_batch_size']
        num_users = self.dataset.user_num

        with torch.no_grad():
            for start_idx in range(1, num_users, batch_size):
                end_idx = min(start_idx + batch_size, num_users)
                user_ids = torch.arange(start_idx, end_idx)

                top_k_results = full_sort_topk(
                    uid_series=user_ids, model=self.model, test_data=self.test_data, k=top_k, device=device
                )
                top_k_scores, top_k_iota = top_k_results

                torch.cuda.empty_cache()

                user_ids_list = user_ids.cpu().numpy()
                top_k_iota_list = top_k_iota.cpu().numpy()

                for i, internal_uid in enumerate(user_ids_list):
                    internal_items = top_k_iota_list[i]

                    original_uid = self.dataset.id2token(self.config['USER_ID_FIELD'], [internal_uid])[0]
                    original_items = self.dataset.id2token(self.config['ITEM_ID_FIELD'], internal_items)

                    recommendations[str(original_uid)] = [int(x) for x in original_items]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(recommendations, f, indent=4)

        print(f'Recommendations save to {save_path}')

    def load_model(self, ckpt_path):
        print(f'Loading model checkpoint from {ckpt_path}')

        checkpoint = torch.load(
            ckpt_path, map_location=self.config['device'], weights_only=False
        )
        self.model.load_state_dict(checkpoint['state_dict'])

        print('Model loaded successfully')
