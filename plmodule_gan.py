#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : solver_gan.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 05.19.2022
# Last Modified Date: 05.19.2022
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>

from ast import Pass
from collections import OrderedDict
from pytorch_lightning import LightningModule
from models_gan import Generator, Discriminator
import torch
import torch.nn.functional as F
from plmodule_data import SparseMolecularDataModule
from mol_utils import all_scores, save_mol_img
import numpy as np
import os


class MolGAN(LightningModule):
    def __init__(self,
                 z_dim,
                 g_conv_dims,
                 d_conv_dims,
                 num_nodes,
                 m_dim,
                 b_dim,
                 dropout_rate,
                 data_module: SparseMolecularDataModule,
                 num_sampled_imgs,
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
        self.data_module = data_module

        # network
        self.G = Generator(conv_dims=g_conv_dims,
                           z_dim=z_dim,
                           vertexes=num_nodes,
                           b_dim=b_dim,
                           m_dim=m_dim,
                           dropout_rate=dropout_rate)
        # TODO(Jing Mai): Why we use `b_dim-1`?
        self.D = Discriminator(conv_dims=d_conv_dims,
                               b_dim=b_dim-1,
                               m_dim=m_dim,
                               dropout_rate=dropout_rate)
        self.V = Discriminator(conv_dims=d_conv_dims,
                               b_dim=b_dim-1,
                               m_dim=m_dim,
                               dropout_rate=dropout_rate)

        self.sampled_img_z = torch.randn(self.hparams.num_sampled_imgs, self.hparams.z_dim)

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        # dynamically adjusted variables
        self.current_lambda_wgan = None

    @staticmethod
    def postprocess(inputs, method, temperature=1.):
        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'softmax':
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]
        else:
            raise ValueError('Unknown postprocessing method: {}'.format(method))

        return [delistify(e) for e in (softmax)]

    def reward(self, mols):
        return self.data_module.reward(mols)

    def matrices2mol(self, node_labels, edge_labels, strict):
        return self.data_module.data.matrices2mol(node_labels, edge_labels, strict)

    def get_reward(self, nodes_hat, edges_hat, method):
        (edges_hard, nodes_hard) = self.postprocess(inputs=(edges_hat, nodes_hat), method=method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        reward = torch.from_numpy(self.reward(mols))
        return reward

    def forward(self, z):
        return self.G(z)

    def compute_gradient_penalty(self, real_edges, real_nodes, fake_edges, fake_nodes):
        """Calculates the gradient penalty loss for WGAN GP"""

        def gp_norm(y, x):
            dydx = torch.autograd.grad(outputs=y, inputs=x,
                                       grad_outputs=torch.ones(y.size()).type_as(y),
                                       create_graph=True, retain_graph=True, only_inputs=True)[0]
            dydx = dydx.view(dydx.size(0), -1)
            return ((dydx.norm(2, dim=1) - 1) ** 2).mean()

        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_edges.size(0), 1, 1, 1).type_as(real_edges)
        # Get random interpolation between real and fake samples
        edge_interpolates = (alpha * real_edges + ((1 - alpha) * fake_edges)).requires_grad_(True)
        node_interpolates = (alpha * real_nodes + ((1 - alpha) * fake_nodes)).requires_grad_(True)

        d_edge_interpolates, d_node_interpolates = self.D(edge_interpolates, None, node_interpolates)

        edge_obj = d_edge_interpolates.mean()
        node_obj = d_node_interpolates.mean()
        edge_gp = gp_norm(edge_obj, edge_interpolates)
        node_gp = gp_norm(node_obj, node_interpolates)
        gradient_penalty = edge_gp + node_gp
        return gradient_penalty

    def on_train_start(self):
        # The first half epochs use the WGAN objective only
        self.current_lambda_wgan = 1

    def on_train_epoch_start(self):
        # The second half epochs using both RL and WGAN.
        if self.current_epoch * 2 >= self.max_epochs:
            self.current_lambda_wgan = self.hparams.lambda_wgan

    def compute_d_loss(self, batch,  batch_idx, z):
        """Computes the discriminator loss for a batch of samples. """
        mols, A_onehot, X_onehot = batch['mols'], batch['A_onehot'], batch['X_onehot']

        # pass real samples to discriminator
        logits_real, features_real = self.D(A_onehot, None, X_onehot)
        # pass latent space samples z to target
        edge_logits, node_logits = self.G(z)
        # postprocess with Gumbel softmax
        (edges_hat, nodes_hat) = self.postprocess(edge_logits, node_logits, method=self.hparams.post_method)
        # pass fake samples to discriminator
        logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)

        # gradient penalty
        grad_penalty = self.compute_gradient_penalty(A_onehot, X_onehot, edges_hat, nodes_hat)

        d_loss_real = torch.mean(logits_real)
        d_loss_fake = torch.mean(logits_fake)
        d_loss = - d_loss_real + d_loss_fake + self.hparams.lambda_gp * grad_penalty

        output = {
            'd_loss': d_loss,
            'd_loss/R': d_loss_real,
            'd_loss/F': d_loss_fake,
            'd_loss/GP': grad_penalty
        }
        return output

    def compute_gv_loss(self, batch,  batch_idx, z):
        """ Computes the generator loss and the value loss for a batch of samples. """
        mols, A_onehot, X_onehot = batch['mols'], batch['A_onehot'], batch['X_onehot']
        # pass latent space samples z to target
        edge_logits, node_logits = self.G(z)
        # postprocess with Gumbel softmax
        edges_hat, nodes_hat = self.postprocess(edge_logits, node_logits, method=self.hparams.post_method)
        # pass fake samples to discriminator
        logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)

        # Value losses
        value_logit_real, _ = self.V(A_onehot, None, X_onehot, activation=torch.sigmoid)
        value_logit_fake, _ = self.V(edges_hat, None, nodes_hat, activation=torch.sigmoid)

        # real reward
        reward_real = torch.from_numpy(self.data_module.reward(mols)).type_as(A_onehot)
        # fake reward
        reward_fake = self.get_reward(nodes_hat, edges_hat)

        g_loss = - logits_fake
        v_loss = (value_logit_real - reward_real) ** 2 + (value_logit_fake - reward_fake) ** 2
        rl_loss = - value_logit_fake

        g_loss = g_loss.mean()
        v_loss = v_loss.mean()
        rl_loss = rl_loss.mean()

        alpha = torch.abs(g_loss.detach() / rl_loss.detach()).detach()
        train_step_G = self.current_lambda_wgan * g_loss + alpha * (1 - self.current_lambda_wgan) * rl_loss
        train_step_V = v_loss

        output = {'g_loss': g_loss,
                  'v_loss': v_loss,
                  'rl_loss': rl_loss,
                  'train_step_G': train_step_G,
                  'train_step_V': train_step_V}

        return output

    def get_scores(self, nodes_logits, edges_logits, post_method):
        mols = self.get_gen_mols(nodes_logits, edges_logits, post_method)
        m0, m1 = all_scores(mols, self.data, norm=True)  # 'mols' is output of Fake Reward
        scores = m1.copy()
        for k, v in m0.items():
            scores[k] = np.array(v)[np.nonzero(v)].mean()
        return scores

    def training_step(self, batch, batch_idx):
        mols, A_onehot, X_onehot = batch['mols'], batch['A_onehot'], batch['X_onehot']
        opt_g, opt_d, opt_v = self.optimizers()

        # ========================================================== #
        #                       Train Discriminator                  #
        # ========================================================== #
        # sample noise
        z = torch.randn(A_onehot.shape[0], self.z_dim).type_as(A_onehot)
        d_loss_dict = self.compute_d_loss(batch, batch_idx, z)
        # back propagate discriminator's gradient if `current_lambda_wgan` is greater than zero.
        if self.current_lambda_wgan > 0:
            opt_d.zero_grad()
            self.manual_backward(d_loss_dict['d_loss'], opt_d)
            opt_d.step()

        # ========================================================== #
        #                Train Generator & Value Network             #
        # ========================================================== #
        gv_loss_dict = self.compute_gv_loss(batch, batch_idx, z)
        # back propagate the generator's and the value network's gradient every `n_critic` steps
        if (self.global_step + 1) % self.hparams.n_critic == 0:
            opt_g.zero_grad()
            opt_v.zero_grad()
            self.manual_backward(gv_loss_dict['train_step_G'], opt_g, retain_graph=True)
            opt_g.step()
            self.manual_backward(gv_loss_dict['train_step_V'], opt_v)
            opt_v.step()

        output = dict(d_loss_dict, **gv_loss_dict)

        return output

    def training_epoch_end(self, outputs):
        keys = outputs[0].keys()
        avg_output = {k: torch.stack([x[k] for x in outputs]).mean().item() for k in keys}
        self.log_dict(avg_output, prefix='train_')

    def _shared_eval_step(self, batch, batch_idx):
        mols, A_onehot, X_onehot = batch['mols'], batch['A_onehot'], batch['X_onehot']
        # sample noise
        z = torch.randn(A_onehot.shape[0], self.z_dim).type_as(A_onehot)
        edge_logits, node_logits = self.G(z)
        d_loss_dict = self.compute_d_loss(batch, batch_idx, z)
        gv_loss_dict = self.compute_gv_loss(batch, batch_idx, z)
        score_dict = self.get_scores(node_logits, edge_logits, self.hparams.post_method)
        metrics = dict(d_loss_dict, **gv_loss_dict, **score_dict)
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        return metrics

    def _shared_eval_epoch_end(self, outputs):
        keys = outputs[0].keys()
        avg_output = {k: torch.stack([x[k] for x in outputs]).mean().item() for k in keys}
        return avg_output

    def validation_epoch_end(self, outputs):
        metrics = self._shared_eval_epoch_end(outputs)
        self.log_dict(metrics, prefix='val_')

    def test_epoch_end(self, outputs):
        metrics = self._shared_eval_epoch_end(outputs)
        self.log_dict(metrics, prefix='test_')

    def configure_optimizers(self):
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=self.hparams.lr_g)
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=self.hparams.lr_d)
        self.opt_v = torch.optim.Adam(self.V.parameters(), lr=self.hparams.lr_v)
        return self.opt_g, self.opt_d, self.opt_v

    def on_epoch_end(self):
        edges_logits, nodes_logits = self.G(self.sampled_img_z)
        mols = self.get_gen_mols(nodes_logits, edges_logits, self.hparams.post_method)
        # Saving molecule images.
        mol_f_name = os.path.join(self.hparams.img_dir_path, 'mol-{}.png'.format(self.current_epoch))
        save_mol_img(mols, mol_f_name, is_test=self.hparams.mode == 'test')
