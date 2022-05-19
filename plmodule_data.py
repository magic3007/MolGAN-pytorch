#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : data_loader.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 05.19.2022
# Last Modified Date: 05.19.2022
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>

import pytorch_lightning
from pytorch_lightning import LightningDataModule

class ChemDataModule(LightningDataModule):
    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims) 