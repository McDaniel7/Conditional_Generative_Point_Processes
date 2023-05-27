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

# Define phi network and fully neural model for PP

class fNet(nn.Module):
    """
    fNet generates random samples according to the probability density function 
    of a point process given the current event `x = (t, m)` and the hidden state
    `h_t` that summarizes the history. 
    """

    def __init__(self, config):
        super(fNet, self).__init__()

        self.batch_size  = config['batch_size']
        self.n_samples   = config['n_samples']
        self.noise_dim   = config['noise_dim']
        self.hid_dim     = config['hid_dim']
        self.data_dim    = config['data_dim']
        # input layer
        self.input_layer = nn.Linear(
            in_features=config['hid_dim'] + config['noise_dim'], 
            out_features=config['mlp_dim'])
        # multiple linear layers
        self.mlp         = nn.ModuleList(
            [nn.Linear(
                in_features=config['mlp_dim'], 
                out_features=config['mlp_dim']) 
            for _ in range(config['mlp_layer']-1)])
        # output layer
        if self.data_dim > 1:
            self.out_layer_time   = nn.Sequential(
                nn.Linear(in_features=config['mlp_dim'], out_features=1), 
                nn.Softplus())
            self.out_layer_loc    = nn.Sequential(
                nn.Linear(in_features=config['mlp_dim'], out_features=config['data_dim']-1)) 
        elif self.data_dim == 1:
            self.out_layer   = nn.Sequential(
                nn.Linear(in_features=config['mlp_dim'], out_features=config['data_dim']), 
                nn.Softplus())
        else:
            raise ValueError("Invalid data dimension")

    def forward(self, h, tn=None):
        """
        input
        h: hidden state                                    [batch_size, hid_dim]
        tn: last event time (optional)               [batch_size, data_dim >= 1]

        output
        xhats: generated samples               [batch_size, n_samples, data_dim]
        """

        batch_size = h.shape[0]
        # generate gaussian noise
        # for earthquake data set to 20.0
        # for 1d synthetic set to 0.2
        # for 3d synthetic set to 0.05
        z = torch.randn(batch_size, self.n_samples, self.noise_dim,
                        device=h.device) * 0.2
        # concatenate generated noise with hidden states
        h = h.unsqueeze(1)                            # [batch_size, 1, hid_dim]
        h = torch.tile(h, (1, self.n_samples, 1))
                                              # [batch_size, n_samples, hid_dim]              
        input = torch.cat([z, h], dim=-1)       
                                  # [batch_size, n_samples, noise_dim + hid_dim]
        # combine first two dimensions
        input = input.view(
            batch_size * self.n_samples, self.noise_dim + self.hid_dim) 
                                 # [batch_size * n_samples, noise_dim + hid_dim]
        out   = torch.tanh(self.input_layer(input))  
                                             # [batch_size * n_samples, hid_dim]
        for layer in self.mlp:
            out = torch.tanh(layer(out))     # [batch_size * n_samples, mlp_dim]
        
        if self.data_dim > 1:
            xhats_time = self.out_layer_time(out)      # [batch_size * n_samples, 1]
            xhats_loc = self.out_layer_loc(out) # [batch_size * n_samples, data_dim-1]
            xhats = torch.concat((xhats_time, xhats_loc), dim=-1)
                                                # [batch_size * n_samples, data_dim]
        elif self.data_dim == 1:            
            xhats = self.out_layer(out)
                                                # [batch_size * n_samples, data_dim]
        else:
            raise ValueError("Invalid data dimension")
            
        # separate first two dimensions
        xhats = xhats.view(batch_size, self.n_samples, self.data_dim)
                                             # [batch_size, n_samples, data_dim]
        return xhats



class NeuralPP(nn.Module):
    """
    """

    def __init__(self, config):
        super(NeuralPP, self).__init__()

        self.batch_size = config['batch_size']
        self.data_dim   = config["data_dim"]
        self.hid_dim    = config["hid_dim"]
        self.seq_len    = config["seq_len"]
        self.kde_bdw    = torch.nn.Parameter(torch.tensor(config["kde_bdw"]), requires_grad=False)

        # init LSTM
        self.lstm       = nn.LSTM(input_size=config['data_dim'],
                                  hidden_size=config['hid_dim'])
        # init fNet
        self.fnet       = fNet(config)
      
    def forward(self, X):
        """
        """
        return 

    def _fKDE(self, dxi, xhats, h=1.):
        """
        fKDE returns the PDF of dxi using KDE.

        input
        dxi: the i-th event (time interval)               [batch_size, data_dim]
        xhats: generated event samples given the hidden state hi
                                             # [batch_size, n_samples, data_dim]
        h: bandwidth
        
        output 

        """
        n_samples = xhats.shape[1]

        # Reflect KDE
        nega_xhats = xhats.clone()
        nega_xhats[:, :, 0] = -xhats[:, :, 0]
        xhats = torch.concat([xhats, nega_xhats], dim=1)
                                           # [batch_size, 2*n_samples, data_dim]
        dxi2 = torch.tile(dxi.unsqueeze(1), (1, 2 * n_samples, 1)) 
                                           # [batch_size, 2*n_samples, data_dim]
        
        ks  = torch.exp(- ((dxi2 - xhats)**2 / h**2).sum(-1) / 2) \
                / np.sqrt(2 * np.pi) ** (self.data_dim) / h.prod()
                                                     # [batch_size, 2*n_samples]
        fi  = ks.sum(1) / n_samples                               # [batch_size]

        return fi    

    
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
    

    def lambda_(self, xs, X, n_sample=100):
        """
        input
        xs: evaluated events                    [batch_size, n_events, data_dim]
        X: sequences of event data               [batch_size, seq_len, data_dim]

        output
        fs:  pdf at evaluated events                                [batch_size]
        lam: lambda at evaluated events                             [batch_size]

        note
        n_events is not seq_len
        """
        n_events = xs.shape[1]
        batch_size, seq_len, _ = X.shape
        # create mask for X in a bid to remove zeros 
        mask     = X[:, :, 0] > 0                        # [batch_size, seq_len]
        # comput hidden states for events in the sequences
        _, _, _, hs = self.log_liklihood(X) # (seq_len+1, [batch_size, hid_dim])
        hs       = torch.permute(hs, (1, 0, 2)) # [batch_size, seq_len+1, hid_dim]
        # generate xhats for KDE
        xhatss    = self.fnet(hs.reshape(batch_size*(seq_len+1), self.hid_dim))
        xhatss    = xhatss.reshape(batch_size, seq_len+1, -1, self.data_dim)
                                   # [batch_size, seq_len+1, n_samples, hid_dim]
        # replace time with interval
        dX       = self._delta_X(X)            # [batch_size, seq_len, data_dim]
        # iteration over n_events
        fs = []
        pbar = tqdm(total = n_events, desc="Calculating fs...")
        for j in range(n_events):
            # the current evaluated events
            xj  = xs[:, j, :]                           # [batch_size, data_dim]

            # find all the events in the past
            xjj = torch.tile(
                xj.unsqueeze(1), 
                (1, self.seq_len, 1))          # [batch_size, seq_len, data_dim]
            Xn  = (X[:, :, 0] < xjj[:, :, 0]) * mask     # [batch_size, seq_len]

            # find the time of the nearest events in the past
            # note: add zero to the head of sequences
            zro = torch.zeros(
                (batch_size, 1, self.data_dim), device=X.device)
                                                     # [batch_size, 1, data_dim]
            Xp0 = torch.cat([zro, X], dim=1) # [batch_size, seq_len+1, data_dim]
            n   = Xn.sum(1)                                       # [batch_size]
            n   = n.unsqueeze(-1).unsqueeze(-1)             # [batch_size, 1, 1]
            n   = torch.tile(n, 
                             (1, 1, self.data_dim))  # [batch_size, 1, data_dim]
            xn  = Xp0.gather(1, n)                   # [batch_size, 1, data_dim]
            # xn  = xn.squeeze()                                  # [batch_size]
            xn  = xn[:, 0, :]                           # [batch_size, data_dim]

            # compute pdf given xj and hn 
            # TODO: dxj only for 1D
            dxj = xj - xn                               # [batch_size, data_dim]
            # - select random samples for xj
            n   = Xn.sum(1)                                       # [batch_size]
            n   = n.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)    
                                                         # [batch_size, 1, 1, 1]
            n   = torch.tile(n, 
                             (1, 1, self.fnet.n_samples, self.data_dim))  
                                          # [batch_size, 1, n_samples, data_dim]
            xhats       = xhatss.gather(1, n)[:, 0, :]   
                                             # [batch_size, n_samples, data_dim]
            # - compute pdf using KDE
            fj          = self._fKDE(dxj, xhats, h=self.kde_bdw)  # [batch_size]
            fs.append(fj)

            pbar.update(1)

        # collection of pdf and lambda of all evaluated events
        fs   = torch.stack(fs, dim=-1)                  # [batch_size, n_events]
        
        return fs


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
        batch_size = X.shape[0]
        mask   = X[:, :, 0] > 0                          # [batch_size, seq_len]
        # replace time with interval
        dX     = self._delta_X(X)
        # init a list for pdf and corresponding hidden states
        fs, hs = [], []
        # xhatss = []
        # init hidden states
        h0     = torch.zeros([1, batch_size, self.hid_dim], device=X.device)
        c0     = torch.zeros([1, batch_size, self.hid_dim], device=X.device)
        h, c   = h0, c0
        hs.append(h.squeeze())
        # lstm update
        out, (_, _) = self.lstm(torch.permute(dX, (1, 0, 2)), (h0, c0))
                                            # out=[seq_len, batch_size, hid_dim]
        hs = torch.concat((h0, out), dim=0)   # [seq_len+1, batch_size, hid_dim]
        # step KDE
        for i in range(5, self.seq_len, 1):    
            dxi         = dX[:, i, :]                   # [batch_size, data_dim]
            dx          = dxi.unsqueeze(0)           # [1, batch_size, data_dim]
            # compute pdf given hn
            hn          = hs[i, :, :]                  # [batch_size, hid_dim]
            # - generate random samples given the last state hn and compute pdf using KDE
            xhats       = self.fnet(hn)
            fi = self._fKDE(dxi, xhats, h=self.kde_bdw)
            # the i-th hidden states and corresponding pdf
            fs.append(fi)
            # xhatss.append(xhats)

        # collection of pdf of all events
        fs = torch.stack(fs, dim=-1)                     # [batch_size, seq_len]
        fs = torch.clamp(fs, min=1e-8)
        mask = mask[:, 5:]
        log_likli = torch.log(fs)[mask].sum()
        fs = fs * mask
        n_events = mask.sum()

        return log_likli, n_events, fs, hs