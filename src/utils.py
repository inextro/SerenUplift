import os
import sys
import torch
import numpy as np
import pandas as pd

LIGHTGCN_CODE_PATH = './lgcn/code'
if LIGHTGCN_CODE_PATH not in sys.path:
    sys.path.insert(0, LIGHTGCN_CODE_PATH)

if len(sys.argv) == 1:
    sys.argv.extend([
        '--dataset', 'movielens', 
        '--layer', '2', 
        '--recdim', '16'
    ])
import world
import model
import dataloader


def get_embeddings():
    world.device = torch.device('cpu')

    print(f'Current Dataset: {world.dataset}')
    print(f'Current emb_dim: {world.config['latent_dim_rec']}')

    dataset = dataloader.Loader(path=os.path.join(world.DATA_PATH, world.dataset))
    rec_model = model.LightGCN(config=world.config, dataset=dataset)
    rec_model = rec_model.to(world.device)

    checkpoint_file = f'{world.model_name}-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar'
    checkpoint_path = os.path.join(world.FILE_PATH, checkpoint_file)
    print(f'Load checkpoints from {checkpoint_path}')

    state_dict = torch.load(checkpoint_path, map_location=world.device)
    rec_model.load_state_dict(state_dict)

    rec_model.eval()
    print('Update Embeddings')
    
    with torch.no_grad():
        user_emb, movie_emb = rec_model.computer()

    return user_emb.numpy(), movie_emb.numpy()


def load_id_map(map_path):
    df = pd.read_csv(map_path, sep=' ')

    return df.set_index('org_id')['remap_id'].to_dict()


if __name__ == '__main__':
    user_emb, movie_emb = get_embeddings()

    print(f'Total number of users: {user_emb.shape[0]}')
    print(f'Total number of movies: {movie_emb.shape[0]}')

    save_path = './embeddings/'
    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, f'user_emb_{user_emb.shape[1]}.npy'), user_emb)
    np.save(os.path.join(save_path, f'movie_emb_{movie_emb.shape[1]}.npy'), movie_emb)