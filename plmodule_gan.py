#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : solver_gan.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 05.19.2022
# Last Modified Date: 05.19.2022
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>

from pytorch_lightning import LightningModule
from models_gan import Generator, Discriminator

class MolGAN(LightningModule):
    def __init__(self, 
                 z_dim,
                 g_conv_dims,
                 d_conv_dims,
                 num_nodes,
                 m_dim,
                 b_dim,
                 dropout_rate,
                 post_method,
                 *args, **kwargs):
        """ MolGAN model.

        Args:
            z_dim (int): sampled latent vector dimension
            g_conv_dims (list): Generator convolutional layer dimensions.
            num_nodes (int): number of nodes in the graph.
            m_dim (int): number of atoms in the molecule
            b_dim (int): number of bonds in the molecule
        """
        super(MolGAN, self).__init__()
        self.save_hyperparameters()
       
        # network
        self.G = Generator(conv_dims=g_conv_dims, 
                           z_dim=z_dim,
                           num_nodes=num_nodes, 
                           b_dim=b_dim,
                           m_dim=m_dim,
                           dropout_rate=dropout_rate)
        import pdb; pdb.set_trace()
        # TODO(Jing Mai): Why we use `b_dim-1`? 
        self.D = Discriminator(conv_dims=d_conv_dims,
                               b_dim=b_dim-1,
                               m_dim=m_dim,
                               dropout_rate=dropout_rate)
        self.V = Discriminator(conv_dims=d_conv_dims,
                               b_dim=b_dim-1,
                               m_dim=m_dim,
                               dropout_rate=dropout_rate)
    
    def forward(self, z):
        pass
    
    def adversarial_loss(self):
        pass
    
    def gp_loss(self):
        pass
    
    def value_loss(self):
        pass
    
    def train_step(self, batch,  batch_idx, optimizer_idx):
        pass
    
    def on_epoch_end(self):
        pass