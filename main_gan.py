#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : main_gan.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 05.19.2022
# Last Modified Date: 05.19.2022
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
import torch
from parse_config import get_GAN_config
import random
import os
import json
import numpy as np
from utils import get_date_str, log_everything

# Code from `PyTorch Lightning`
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    # make all the path-like strings absolute
    if config.desc_ap_file is not None:
        config.desc_ap_file = os.path.abspath(config.desc_ap_file)

    if config.desc_ap_file is not None:
        with open(config.desc_ap_file, 'a') as fp:
            fp.write(config.result_dir + ": " + config.desc + '\n')

    if config.mode == 'train':
        with open(os.path.join(config.log_dir, 'json_cli_config.json'), 'w') as fp:
            json.dump(vars(config), fp=fp, indent=4)

    # Miscellaneous
    seed_everything(config.seed)

    # Logger
    # https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
    log_file_name = os.path.join(config.log_dir, config.abbr + '.log')
    log_everything(log_file_name)


if __name__ == '__main__':
    config = get_GAN_config()
    print(config)
    main(config)
