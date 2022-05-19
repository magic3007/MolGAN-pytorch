#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : models_gan.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 05.19.2022
# Last Modified Date: 05.19.2022
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dims, z_dim, num_nodes, b_dim, m_dim, dropout_rate=0.):
        super(Generator, self).__init__()
        
class Discriminator(nn.Module):
    """Discriminator network."""

    def __init__(self, conv_dims, b_dim, m_dim,with_features=False, f_dim=0, dropout_rate=0.):
        super(Discriminator, self).__init__()
    