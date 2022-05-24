#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : data_loader.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 05.19.2022
# Last Modified Date: 05.19.2022
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>

from pytorch_lightning import LightningDataModule
from data.sparse_molecular_dataset import SparseMolecularDataset
import torch
from torch_utils import label2onehot, DictlikeDataset
from torch.utils.data import DataLoader
from mol_utils import MolecularMetrics
import numpy as np

def gen_data_dict(data:SparseMolecularDataset, idx):
    data_dict = {
            "mols": data.data[idx],
            "smlie": data.smiles[idx],
            "S": data.data_S[idx],
            "A": torch.from_numpy(data.data_A[idx]).long(),
            "X": torch.from_numpy(data.data_X[idx]).long(),
            "D": data.data_D[idx],
            "F": data.data_F[idx],
            "Le": data.data_Le[idx],
            "Lv": data.data_Lv[idx],
    }
    data_dict["A_onehot"] =label2onehot(data_dict["A"], data.bond_num_types)
    data_dict["X_onehot"] = label2onehot(data_dict["X"], data.atom_num_types)
    return data_dict
    
def all_scores(mols, data, norm=False, reconstruction=False):
    m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
        'NP': MolecularMetrics.natural_product_scores(mols, norm=norm),
        'QED': MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
        'Solute': MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=norm),
        'SA': MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=norm),
        'diverse': MolecularMetrics.diversity_scores(mols, data),
        'drugcand': MolecularMetrics.drugcandidate_scores(mols, data)}.items()}

    m1 = {'valid': MolecularMetrics.valid_total_score(mols) * 100,
          'unique': MolecularMetrics.unique_total_score(mols) * 100,
          'novel': MolecularMetrics.novel_total_score(mols, data) * 100}

    return m0, m1

class SparseMolecularDataModule(LightningDataModule):
    def __init__(self, data: SparseMolecularDataset, batch_size: int, num_workers: int, 
                 metric: str,
                 shuffle: bool = True,
                     *args, **kwargs):
        super(SparseMolecularDataModule, self).__init__() 
        self.save_hyperparameters(ignore=['data'])
        self.data = data
        self.dims = (len(data), data.vertexes, data.atom_num_types, data.bond_num_types)
    
    def __len__(self):
        """ Return number of samples in the dataset. """
        return self.dims[0]
   
     
    @property
    def vertexes(self):
        return self.dims[1]
    
    @property
    def atom_num_types(self):
        return self.dims[2]
    
    @property
    def bond_num_types(self):
        """ Return number of bond types in the dataset. Note that Bond Type #0 represents the absence of bond. """ 
        return self.dims[3]
        
    def reward(self, mols):
        rr = 1.
        for m in ('logp,sas,qed,unique' if self.hparams.metric == 'all' else self.hparams.metric).split(','):
            if m == 'np':
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == 'logp':
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == 'sas':
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, self.data)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, self.data)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, self.data)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))

        return rr.reshape(-1, 1)

    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            self.train_dictlike_data = DictlikeDataset(
                gen_data_dict(self.data, self.data.train_idx),
                len(self.data.train_idx))
            self.val_dictlike_data = DictlikeDataset(
                gen_data_dict(self.data, self.data.validation_idx),
                len(self.data.validation_idx))
        if stage == "test" or stage is None:
            self.test_dictlike_data = DictlikeDataset(
                gen_data_dict(self.data, self.data.test_idx),
                len(self.data.test_idx))
            
    def train_dataloader(self):
        return DataLoader(self.train_dictlike_data, 
                          collate_fn=self.train_dictlike_data.collate_fn,
                          batch_size=self.hparams.batch_size, shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers) 

    def val_dataloader(self):
        return DataLoader(self.val_dictlike_data, 
                          collate_fn=self.val_dictlike_data.collate_fn,
                          batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)    
   
    def test_dataloader(self):
        return DataLoader(self.test_dictlike_data, 
                          collate_fn=self.test_dictlike_data.collate_fn,
                          batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)    
    
if __name__ == '__main__':
    pass
    