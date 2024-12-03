#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats
from matplotlib import animation
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)
import os
import sys
import arrow
from itertools import product
import random

import torch
from torch import nn
from torch.autograd import grad
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.Softplus(beta=100),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextFCnet(nn.Module):
    def __init__(self, data_dim, hid_dim, input_dim, mlp_dims):
        super(ContextFCnet, self).__init__()

        self.data_dim  = data_dim
        self.hid_dim  = hid_dim
        self.input_dim = input_dim
        
        mlp_dims_tuple = tuple(map(int, mlp_dims.split("-")))
        self.layer_dims_in = (input_dim,) + mlp_dims_tuple
        self.layer_dims_out = mlp_dims_tuple + (data_dim,)

        self.timeembed = EmbedFC(1, input_dim)
        self.contextembed = EmbedFC(hid_dim, input_dim)
        self.pointembed = EmbedFC(data_dim, input_dim)

        Layers = []
        for layer_in, layer_out in zip(self.layer_dims_in, self.layer_dims_out):
            Layers.append(nn.Linear(layer_in, layer_out))
            if layer_out != data_dim:
                Layers.append(nn.Softplus(beta=100))
        self.layers = nn.Sequential(*Layers)
        
#         self.out1   = nn.Linear(self.layer_dims_in[-1], 1)
#         self.out2   = nn.Linear(self.layer_dims_in[-1], self.data_dim - 1)
        
#         self.softplus = nn.Softplus(beta=100)
#         self.relu     = nn.ReLU()
#         self.sigmoid  = nn.Sigmoid()
        
        
    def forward(self, x, h, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.pointembed(x)

        # mask out context if context_mask == 1
        # context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.hid_dim)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        h = h * context_mask
        
        # embed context, time step
        hemb = self.contextembed(h).view(-1, self.input_dim)
        temb = self.timeembed(t).view(-1, self.input_dim)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)
        
        out = self.layers(hemb*x + temb)
        
        return out

# DDPM

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, config):
        super(DDPM, self).__init__()
        self.nn_model = ContextFCnet(config["data_dim"], config["hid_dim"],
                                     config["input_dim"], config["mlp_dims"])

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(config["beta1"], config["beta2"], config["n_T"]).items():
            self.register_buffer(k, v)

        self.n_T = config["n_T"]
        self.device = config["device"]
        self.drop_prob = config["drop_prob"]
        self.loss_mse = nn.MSELoss(reduction="none")

    def forward(self, x, h):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None] * x
            + self.sqrtmab[_ts, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(h[..., [0]])+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, h, _ts[..., None] / self.n_T, context_mask)).sum(-1)

    
    def sample(self, h, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        
        # h: given context, history encoder, [ n_sample, hid_dim ]

        n_sample = h.shape[0]
        x_i = torch.randn(n_sample, self.nn_model.data_dim).to(device)  # x_T ~ N(0, 1), sample initial noise
        h_i = h.to(device) # context for us just cycles throught the mnist labels

        # don't drop context at test time
        context_mask = torch.zeros_like(h_i[..., [0]]).to(device)
        
        # double the batch
        h_i = h_i.repeat(2, 1)
        context_mask = context_mask.repeat(2, 1)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1)
            
            # double batch
            x_i = x_i.repeat(2,1)
            t_is = t_is.repeat(2,1)

            z = torch.randn(n_sample, self.nn_model.data_dim).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, h_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


class NeuralPP(nn.Module):
    """
    """

    def __init__(self, config):
        super(NeuralPP, self).__init__()

        # self.batch_size = config['batch_size']
        # self.seq_len    = config["seq_len"]
        self.data_dim   = config["data_dim"]
        self.hid_dim    = config["hid_dim"]

        # init LSTM
        self.lstm       = nn.LSTM(input_size=config['data_dim'],
                                  hidden_size=config['hid_dim'])
        # init fNet
        self.ddpm       = DDPM(config)

    def forward(self, X):
        """
        """
        return


    def _delta_X(self, X):
        """
        input
        X: sequences of event data               [batch_size, seq_len, data_dim]

        output
        dX: sequences of event data (time has been replaced by interval)
                                                 [batch_size, seq_len, data_dim]
        """
        batch_size = X.shape[0]
        mask = X[:, :, 0] > 0                            # [batch_size, seq_len]
        mask = torch.tile(
            mask.unsqueeze(-1),
            (1, 1, self.data_dim))             # [batch_size, seq_len, data_dim]
        zro  = torch.zeros(
                (batch_size, 1, self.data_dim), device=X.device)
                                                     # [batch_size, 1, data_dim]
        Xp0  = torch.cat([zro, X], dim=1)    # [batch_size, seq_len+1, data_dim]
        dt   = (X[:, :, 0] - Xp0[:, :-1, 0]).unsqueeze(-1)
                                                      # [batch_size, seq_len, 1]
        dX   = torch.cat(
            [dt, X[:, :, 1:]], dim=-1)         # [batch_size, seq_len, data_dim]
        dX   = dX * mask
        return dX
    
    
    def _transform_X(self, X):
        
        dX   = self._delta_X(X)
        
        dX[:, :, 0] = torch.where((X[:, :, 0] > 0) * (X[:, :, 0] * 100 <= 20), torch.log(torch.exp(dX[:, :, 0] * 100.) - 1) / 100., dX[:, :, 0])
#         dX[:, :, 1:] = torch.where((X[:, :, 0] > 0)[:, :, None], torch.tan(dX[:, :, 1:] * 2 - 1), 0.)
#         dX[:, :, 1:3] = torch.where((X[:, :, 0] > 0)[:, :, None], 0.5*torch.log((dX[:, :, 1:3] * 2)/(2 - dX[:, :, 1:3] * 2)), 0.)
        
        return dX
    
    
    def _detransform_X(self, X):
        
        dtX = torch.zeros_like(X)
#         dtX[..., 1:] = (torch.tanh(X[..., 1:]) + 1) / 2.
        dtX[..., 0]  = F.softplus(X[..., 0], beta=100)
        dtX[..., 0]  = torch.cumsum(dtX[..., 0], dim=1)
        
        return dtX
    

    def log_liklihood(self, X):
        """
        input
        X: sequences of event data               [batch_size, seq_len, data_dim]

        output
        log_likli: average log likelihood value                           scalar
        fs: pdf of all events (padded by zeros)            [batch_size, seq_len]
        hs: hidden states of all events         (seq_len, [batch_size, hid_dim])

        reference:
        note - to get individual hidden states, you have to indeed loop over for
               each individual timestep and collect the hidden states.
        https://discuss.pytorch.org/t/how-to-retrieve-hidden-states-for-all-time-steps-in-lstm-or-bilstm/1087
        """
        # create mask for X in a bid to remove zeros when calculating pdf
        batch_size, seq_len, _ = X.shape
        mask   = X[:, :, 0] > 0                          # [batch_size, seq_len]
        # replace time with interval
        dX_trans  = self._transform_X(X)
        # init hidden states
        h0     = torch.zeros([1, batch_size, self.hid_dim], device=X.device)
        c0     = torch.zeros([1, batch_size, self.hid_dim], device=X.device)
        # lstm update
        out, (_, _) = self.lstm(torch.permute(dX_trans, (1, 0, 2)), (h0, c0))
                                            # out=[seq_len, batch_size, hid_dim]
        hs = torch.concat((h0, out), dim=0)   # [seq_len+1, batch_size, hid_dim]

        losses = []
        # step DDPM
        for i in range(0, seq_len, 1):
            dxi         = dX_trans[:, i, :]              # [batch_size, data_dim]
            # compute pdf given hn
            hn          = hs[i, :, :]                    # [batch_size, hid_dim]
            # - generate random samples given the last state hn and compute pdf using KDE
            loss        = self.ddpm(dxi, hn)             # [batch_size]
            # the i-th hidden states and corresponding pdf
            losses.append(loss)

        # collection of pdf of all events
        losses = torch.stack(losses, dim=-1)             # [batch_size, seq_len]
        mask = mask[:, :]
        n_events = mask.sum()
        loss_total = losses[mask].sum() / n_events

        # print("KDE samples: Min: %.5f, Mean: %.5f, Max: %.5f" % (torch.min(xhatss), torch.mean(xhatss), torch.max(xhatss)))

        return loss_total, n_events, hs
