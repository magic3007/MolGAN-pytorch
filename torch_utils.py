#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : torch_utils.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 05.19.2022
# Last Modified Date: 05.19.2022
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
import random
import os
from typing import KeysView
import torch
import numpy as np
from torch.utils.data import Dataset


def seed_everything(seed=42):
    # Code from `PyTorch Lightning`
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label2onehot(labels: torch.Tensor, dim):
    """Convert label indices to one-hot vectors."""
    out = torch.zeros(list(labels.size()) + [dim]).to(labels.device)
    out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
    return out


class DictlikeDataset(Dataset):
    """ Dataset that wraps a dicts of lists with the same length.
        Typically, you should also pass the static method `collate_fn` to the dataloader to merge a list of samples to batch.
    """

    def __init__(self, data_dict: dict, len):
        self.data_dict = data_dict
        self.len = len
        self.data_list = [{key: value[i] for key, value in data_dict.items()} for i in range(len)]

    @property
    def keys(self):
        return self.data_dict.keys()
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_list[index]

    @staticmethod
    def collate_fn(batch_list:list):
        """
        Args:
            batch_list (list): A list for a batch of samples.
        """
        def stack(x_list):
            if isinstance(x_list[0], np.ndarray):
                return np.stack(x_list, 0)
            elif isinstance(x_list[0], torch.Tensor):
                return torch.stack(x_list, 0)
            else:
                return x_list
        keys = batch_list[0].keys() 
        batch_dict = {
            key : stack([d[key] for d in batch_list]) for key in keys
        }
        return batch_dict
        