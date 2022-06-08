#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : args.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 05.19.2022
# Last Modified Date: 05.19.2022
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
import argparse
import os
import sys
import json


def get_GAN_config():
    program_name = os.path.basename(sys.argv[0])
    config_parser = argparse.ArgumentParser(
        prog=program_name, add_help=False)
    # JSON support
    config_parser.add_argument('--config', type=str, default='config.json', help='Configuration JSON file')
    parser = argparse.ArgumentParser(
        parents=[config_parser], description="Train or test MolGAN. Note that the command-line options will override the config file.")

    # Miscellaneous
    parser.add_argument('--desc', type=str, help='Description of the experiment.')
    parser.add_argument('--desc_ap_file', type=str, help='If set, the description will be appended to the file.')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='train or test')

    # Model Config

    # Data Config

    # Training or Testing Config
    parser.add_argument('--lambda_wgan', type=float, help='weight of the wgan loss')
    parser.add_argument('--resume_ckpt_path', type=str, help='resume training from the checkpoint')

    # Result Config
    parser.add_argument('--result_dir', type=str, help='Directory to save results')

    args, left_argv = config_parser.parse_known_args()
    if args.config is not None:
        json_dict = json.load(open(args.config))
        for key, value in json_dict.items():
            setattr(args, key, value)

    # Load command-line options which overwrite the JSON file
    parser.parse_args(left_argv, namespace=args)
    return args


if __name__ == '__main__':
    args = get_GAN_config()
    print("Config: ", args)
