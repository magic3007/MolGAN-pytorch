#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : torch_utils.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 05.19.2022
# Last Modified Date: 05.19.2022
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
import random
import os
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
    def __init__(self, data:dict, len):
        self.data = data
        self.len = len
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return {
            key : value[index] for key, value in self.data.items()
        } 
