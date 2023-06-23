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

# Backup

class CVAE(nn.Module):
    """
    """
    def __init__(self, config):
        super(CVAE, self).__init__()

        self.data_dim   = config['data_dim']
        self.hid_dim    = config["hid_dim"]
        self.mlp_dim    = config["mlp_dim"]
        self.noise_dim  = config["noise_dim"]
        self.data_cat_hid = self.data_dim + self.hid_dim
        
        self.fc1  = nn.Linear(self.data_cat_hid, self.mlp_dim)
        self.encode_mlp   = nn.ModuleList(
            [nn.Linear(
                in_features=config['mlp_dim'],
                out_features=config['mlp_dim'])
            for _ in range(config['mlp_layer']-1)])
        self.fc21 = nn.Linear(self.mlp_dim, self.noise_dim)
        self.fc22 = nn.Linear(self.mlp_dim, self.noise_dim)

        self.fc3  = nn.Linear(self.noise_dim + self.hid_dim, self.mlp_dim)

        self.decode_mlp   = nn.ModuleList(
            [nn.Linear(
                in_features=config['mlp_dim'],
                out_features=config['mlp_dim'])
            for _ in range(config['mlp_layer']-1)])
        self.fc41 = nn.Linear(self.mlp_dim, 1)
        self.fc42 = nn.Linear(self.mlp_dim, self.data_dim - 1)

        self.softplus = nn.Softplus()
        self.relu     = nn.ReLU()

        self.fc22.apply(self.init_weights)

    def encode(self, x, h):
        x = x.view(-1, self.data_dim)
        x = torch.cat((x, h), 1)
        x = self.softplus(self.fc1(x))
        return self.fc21(x), self.fc22(x)

    def decode(self, z, h):
        z = z.view(-1, self.noise_dim)
        z = torch.cat((z, h), 1)
        z = self.softplus(self.fc3(z))
        return torch.cat((self.relu(self.fc41(z)), self.fc42(z)), -1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, h):
        mu, logvar = self.encode(x, h)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, h)
        return recon_x, mu, logvar

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.1)
        

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
        self.cvae       = CVAE(config)
      
    def forward(self, X):
        """
        """
        return   

    
    def _vae_loss_function(self, x, recon_x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction="none").sum(-1)                      # [batch_size]
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                                                                  # [batch_size]
        return BCE, KLD

    
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
        dX     = self._delta_X(X)
        # init hidden states
        h0     = torch.zeros([1, batch_size, self.hid_dim], device=X.device)
        c0     = torch.zeros([1, batch_size, self.hid_dim], device=X.device)
        # lstm update
        out, (_, _) = self.lstm(torch.permute(dX, (1, 0, 2)), (h0, c0))
                                            # out=[seq_len, batch_size, hid_dim]
        hs = torch.concat((h0, out), dim=0)   # [seq_len+1, batch_size, hid_dim]

        BCEs, KLDs = [], []
        # step CVAE
        for i in range(5, seq_len, 1):    
            dxi         = dX[:, i, :]                   # [batch_size, data_dim]
            # compute pdf given hn
            hn          = hs[i, :, :]                    # [batch_size, hid_dim]
            # - generate random samples given the last state hn and compute pdf using KDE
            recon_x,mu,logvar   = self.cvae(dxi, hn)
            BCE, KLD = self._vae_loss_function(dxi, recon_x, mu, logvar)
            # the i-th hidden states and corresponding pdf
            BCEs.append(BCE)
            KLDs.append(KLD)

        # collection of pdf of all events
        BCEs = torch.stack(BCEs, dim=-1)                 # [batch_size, seq_len]
        KLDs = torch.stack(KLDs, dim=-1)                 # [batch_size, seq_len]
        mask = mask[:, 5:]
        n_events = mask.sum()
        loss = BCEs[mask].sum() / n_events / self.cvae.data_dim + \
                KLDs[mask].sum() / n_events / self.cvae.noise_dim

        # print("KDE samples: Min: %.5f, Mean: %.5f, Max: %.5f" % (torch.min(xhatss), torch.mean(xhatss), torch.max(xhatss)))

        return loss, n_events, hs