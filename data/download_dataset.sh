#!/usr/bin/env bash
# File              : download_dataset.sh
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 05.19.2022
# Last Modified Date: 05.19.2022
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
# Credit to https://github.com/nicola-decao/MolGAN/blob/master/data/download_dataset.sh

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz
tar xvzf gdb9.tar.gz
rm gdb9.tar.gz

wget https://github.com/gablg1/ORGAN/raw/master/organ/NP_score.pkl.gz
wget https://github.com/gablg1/ORGAN/raw/master/organ/SA_score.pkl.gz
