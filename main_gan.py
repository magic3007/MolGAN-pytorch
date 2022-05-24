#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : main_gan.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 05.19.2022
# Last Modified Date: 05.19.2022
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
from parse_config import get_GAN_config
import os
import json
from utils import get_date_str, log_everything
import pytorch_lightning
import logging
from plmodule_gan import MolGAN
from data.sparse_molecular_dataset import SparseMolecularDataset
from plmodule_data import SparseMolecularDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

log = logging.getLogger(__name__)


def main(config):
    # purify the config
    if config.mode == 'train':
        config.result_dir = os.path.abspath(os.path.join(config.result_dir, get_date_str()))
        config.ckpt_dir = os.path.join(config.result_dir, 'ckpt')
        config.log_dir = os.path.join(config.result_dir, 'log')
        config.img_dir = os.path.join(config.result_dir, 'img')
    elif config.mode == 'test':
        a_test_time = get_date_str()
        config.result_dir = os.path.abspath(config.result_dir)
        config.ckpt_dir = os.path.join(config.result_dir, 'ckpt')
        config.log_dir = os.path.join(config.result_dir, 'post_test', a_test_time, 'log')
        config.img_dir = os.path.join(config.result_dir, 'post_test', a_test_time, 'img')
    else:
        raise ValueError('Unknown mode: {}'.format(config.mode))

    for dir in [config.result_dir, config.log_dir, config.ckpt_dir, config.img_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    if config.desc_ap_file is not None:
        config.desc_ap_file = os.path.abspath(config.desc_ap_file)

    config.log_file_name = os.path.join(config.log_dir, config.abbr + '.log')

    # append the description to the file
    if config.desc_ap_file is not None:
        with open(config.desc_ap_file, 'a') as fp:
            fp.write(config.result_dir + ": " + config.desc + '\n')

    # save the cli overwritten config
    if config.mode == 'train':
        with open(os.path.join(config.log_dir, 'json_cli_config.json'), 'w') as fp:
            json.dump(vars(config), fp=fp, indent=4)

    # setup logging to both file and stderr and set the logging format
    log_everything(config.log_file_name)

    # Miscellaneous
    pytorch_lightning.seed_everything(config.seed)

    data = SparseMolecularDataset()
    data.load(config.data_dir)
    dm = SparseMolecularDataModule(data=data,
                                   batch_size=config.batch_size,
                                   num_workers=config.num_workers,
                                   metric=config.metric)
    model = MolGAN(
        num_nodes=dm.vertexes,
        m_dim=dm.atom_num_types,
        b_dim=dm.bond_num_types,
        data_module=dm,
        **vars(config))
    checkpoint_callback = ModelCheckpoint(dirpath=config.ckpt_dir)
    trainer = Trainer(callbacks=[checkpoint_callback])

    if config.mode == 'train':
        trainer.fit(model, datamodule=dm,
                    devices=config.devices,
                    accelerator=config.accelerator,
                    ckpt_path=config.resume_ckpt_path,
                    check_val_every_n_epoch=config.check_val_every_n_epoch)
    elif config.mode == 'test':
        trainer.test(model, datamodule=dm,
                     devices=config.devices,
                     accelerator=config.accelerator,
                     ckpt_path=config.resume_ckpt_path)
                

if __name__ == '__main__':
    config = get_GAN_config()
    print(config)
    main(config)
