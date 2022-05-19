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
from plmodule_data import ChemDataModule
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

    # config feasibility check
    assert config.device == 'cpu' or torch.cuda.is_available(), 'GPU is not available'

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

    # TODO(Jing Mai): add Pytorch Lightning Data Module
    # dm = ChemDataModule()
    # model = MolGAN(
    #     num_nodes=dm.num_nodes,
    #     m_dim=dm.atom_num_types,
    #     b_dim=dm.bond_num_types,
    #     **vars(config))


if __name__ == '__main__':
    config = get_GAN_config()
    print(config)
    main(config)
