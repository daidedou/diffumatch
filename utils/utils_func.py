import os
import os.path as osp
import scipy.sparse as sp
import shutil
from pathlib import Path
import torch
import numpy as np
import re
import os
import requests

def ensure_pretrained_file(hf_url: str, save_dir: str = "pretrained", filename: str = "pretrained.pkl", token: str = None):
    """
    Ensure that a pretrained file exists locally.
    If the folder is empty, download from Hugging Face.

    Args:
        hf_url (str): Hugging Face file URL (resolve/main/...).
        save_dir (str): Directory to store pretrained file.
        filename (str): Name of file to save.
        token (str): Optional Hugging Face token (for gated/private repos).
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        print(f"✅ Found pretrained file: {save_path}")
        return save_path

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    print(f"⬇️ Downloading pretrained file from {hf_url} to {save_path} ...")
    response = requests.get(hf_url, headers=headers, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("✅ Download complete.")
    return save_path


def may_create_folder(folder_path):
    if not osp.exists(folder_path):
        oldmask = os.umask(000)
        os.makedirs(folder_path, mode=0o777)
        os.umask(oldmask)
        return True
    return False


def make_clean_folder(folder_path):
    success = may_create_folder(folder_path)
    if not success:
        shutil.rmtree(folder_path)
        may_create_folder(folder_path)

def str_delta(delta):
    s = delta.total_seconds()
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

def desc_from_config(config, world_size):
    cond_str = 'uncond' #'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if config["perfs"]["fp16"] else 'fp32'
    name_data = config["data"]["name"]
    if "abs" in config["data"]:
        name_data += "abs" if config["data"]["abs"] else ""
    desc = (f'{name_data:s}-{cond_str:s}-{config["architecture"]["model"]:s}-'
            f'gpus{world_size:d}-batch{config["hyper_params"]["batch_size"]:d}-{dtype_str:s}')
    if config["misc"]["precond"]:
        desc += f'-{config["misc"]["precond"]:s}'

    if "desc" in config["misc"]:
        if config["misc"]["desc"] is not None:
            desc += f'-{config["misc"]["desc"]}'
    return desc

def get_dataset_path(config):
    name_exp = config["data"]["name"]
    if "add_name" in config:
        if config["add_name"]["do"]:
            name_exp += "_" + config["add_name"]["name"]
    dataset_path = os.path.join(config["data"]["root_dir"], name_exp)
    return name_exp, dataset_path

def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape), dtype=torch.float32).coalesce()

def convert_dict(np_dict, device="cpu"):
    torch_dict = {}
    for k, value in np_dict.items():
        if sp.issparse(value):
            torch_dict[k] = sparse_np_to_torch(value).to(device)
            if torch_dict[k].dtype == torch.int32:
                torch_dict[k] = torch_dict[k].long().to(device)
        elif isinstance(value, np.ndarray):
            torch_dict[k] = torch.from_numpy(value).to(device)
            if torch_dict[k].dtype == torch.int32:
                torch_dict[k] = torch_dict[k].squeeze().long().to(device)
        else:
            torch_dict[k] = value
    return torch_dict


def convert_dict_torch(in_dict, device="cpu"):
    torch_dict = {}
    for k, value in in_dict.items():
        if isinstance(value, torch.Tensor):
            torch_dict[k] = in_dict[k].to(device)
    return torch_dict

def batchify_dict(torch_dict):
    for k, value in torch_dict.items():
        if isinstance(value, torch.Tensor):
            if torch_dict[k].dtype != torch.int64:
                torch_dict[k] = torch_dict[k].unsqueeze(0)
