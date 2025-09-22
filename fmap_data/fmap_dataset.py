import os
import lmdb
import numpy as np
import torch
import io
import scipy
from itertools import permutations
from pathlib import Path
import re

def p2p_to_FM(p2p, eigvects1, eigvects2, A2=None):
    if A2 is not None:
        if A2.shape[0] != eigvects2.shape[0]:
            raise ValueError("Can't compute pseudo inverse with subsampled eigenvectors")

        if len(A2.shape) == 1:
            return eigvects2.T @ (A2[:, None] * eigvects1[p2p, :])

        return eigvects2.T @ A2 @ eigvects1[p2p, :]

    return scipy.linalg.lstsq(eigvects2, eigvects1[p2p, :])[0]

def get_2d_grid_from_lambda(lam, cfg):
    """
    lam: torch Tensor (b_size, n_fmap, 2) containing eigenvalues of first and second shape
    return:
    gird: (n, 1, grid_size, grid_size)
    """
    if cfg["lambda_pos"] == "diff":
        grid = lam[:, 0, None] - lam[None, :, 1]
    elif cfg["lambda_pos"] == "lambda":
        grid_size = lam.shape[-2]
        grid_x = lam[:, 0]
        grid_y = lam[:, 1]
        grid = torch.stack([grid_x.unsqueeze(-1).expand(M, -1, grid_y.shape[1]),
                grid_y.unsqueeze(1).expand(M, grid_x.shape[1], -1)],
                dim=-1) ## generalized version of meshgrid 
    # https://stackoverflow.com/questions/72782751/how-to-perform-torch-meshgrid-over-multiple-tensors-in-parallel
        grid = grid.reshape([2, 1, grid_size, grid_size])
    return grid


def contains_any_regex(substrings, texts):
    pattern = re.compile('|'.join(map(re.escape, substrings)))  # Compile regex once
    return [text for text in texts if bool(pattern.search(text))]  # Apply to all texts efficiently


TRAIN_IDX = sids  =['50002', '50004', '50007', '50009', '50020',
    '50021', '50022', '50025']
TEST_IDX = ['50026', '50027']

class FmapLiveTemplateDataset(torch.utils.data.Dataset):

    def __init__(self, cache_path, absolute=False, train=None):
        self.root_path = cache_path
        file_list = [p for p in Path(cache_path).rglob('*.npz')]
        if train is None:
            self.files = file_list
        elif train:
            self.files = contains_any_regex(TRAIN_IDX, file_list)
        else:
            self.files = contains_any_regex(TEST_IDX, file_list)
        n_fmaps = 2*len(self.files)
        self.real_data_len = n_fmaps
        cache_template = np.load(os.path.join(cache_path, "template_clean.npz"))
        self.evecs_template = cache_template['evecs']
        self.evals_template = cache_template['evals']
        self.mass_template = cache_template['mass']
        self.n_v = cache_template["vertices"].shape[0]
        self.inD = np.arange(self.n_v)
        self.abs = absolute
    
    def __getitem__(self, idx, n_fmap=30):
        file_name = self.files[idx//2]
        dict_file = np.load(file_name)
        if idx %2 == 0:
            fmap_ = p2p_to_FM(self.inD, self.evecs_template,
                              dict_file['evecs'], dict_file['mass'])
        else:
            fmap_ = p2p_to_FM(self.inD, dict_file['evecs'],
                              self.evecs_template, self.mass_template)
        if self.abs:
            fmap_ = np.abs(fmap_)
        return fmap_[:n_fmap, :n_fmap].reshape((1, n_fmap, n_fmap))

    def __len__(self, ):
        return self.real_data_len


class FmapLivePairDataset(torch.utils.data.Dataset):
    def __init__(self, cache_path, img=False, absolute=False):
        self.root_path = cache_path
        self.files = [p for p in Path(self.root_path).rglob('*.npz')]
        self.combs = list(permutations(self.files, 2))
        n_fmaps = 2*len(os.listdir(self.root_path))
        self.real_data_len = n_fmaps
        self.cache_template = np.load(os.path.join(self.root_path, self.files[0]))
        self.n_v = self.cache_template["vertices"].shape[0]
        self.inD = np.arange(self.n_v)
        self.abs = absolute
    
    def __getitem__(self, idx, n_fmap=30):
        file_name_1, file_name_2 = self.combs[idx]
        dict_1 = np.load(os.path.join(self.root_path, file_name_1))
        dict_2 = np.load(os.path.join(self.root_path, file_name_2))
        fmap_ = p2p_to_FM(self.inD, dict_1['evecs'], dict_2['evecs'], dict_2['mass'])
        lamb = np.concatenate((dict_1['evals'][:, None], dict_2['evals'][:, None]), axis=-1)
        if self.abs:
            fmap_ = np.abs(fmap_)
        return fmap_[:n_fmap, :n_fmap].reshape((1, n_fmap, n_fmap)), lamb[:n_fmap]

    def __len__(self, ):
        return self.real_data_len


class FmapNPDataset(torch.utils.data.Dataset):
    def __init__(self, npy_path, min_max=False, img=False, absolute=False):
        self.img = img 
        self.root_path = npy_path
        self.fmaps = np.load(self.root_path)
        self.shape = self.fmaps.shape[-1]
        self.real_data_len = len(self.fmaps)
        self.min_max = min_max
        self.abs = absolute
        print("DATA LOADED")

    def __getitem__(self, idx):
        """
        Return:
            [21, 3] or [21, 6] for poses including body and root orient
            [10] for shapes (betas)  [Optimal]
        """
        fmap = self.fmaps[idx]
        if self.abs:
            fmap = np.abs(fmap)
        return fmap.reshape((1, self.shape, self.shape))

    def __len__(self, ):
        return self.real_data_len