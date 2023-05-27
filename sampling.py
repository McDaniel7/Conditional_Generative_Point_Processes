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


def KDE_NPP_data_generator(NPP, batch_size, seq_len):
    """
    Efficient generative point process based on non-parametric learning
    """
    old_n_sample = NPP.fnet.n_samples
    NPP.fnet.n_samples = 1
    points = []
    # init hidden states
    h0     = torch.zeros([1, batch_size, NPP.hid_dim])
    c0     = torch.zeros([1, batch_size, NPP.hid_dim])
    h, c   = h0, c0
    # generation step
    with torch.no_grad():
        for i in tqdm(range(seq_len)):
            xhats = NPP.fnet(h.squeeze())          # [batch_size, 1, data_dim]
            points.append(xhats.squeeze())
            out, (h, c) = NPP.lstm(torch.permute(xhats, (1, 0, 2)), (h, c)) 
        
    # collection of events
    points = torch.stack(points, dim=1)
    points[:, :, 0] = torch.cumsum(points[:, :, 0], dim=1)
    NPP.fnet.n_samples = old_n_sample

    return points.cpu().numpy()                # [batch_size, seq_len, data_dim]


def VAE_NPP_data_generator(NPP, batch_size, seq_len):
    points = []
    # init hidden states
    h0     = torch.zeros([1, batch_size, NPP.hid_dim])
    c0     = torch.zeros([1, batch_size, NPP.hid_dim])
    h, c   = h0, c0
    # generation step
    with torch.no_grad():
        for i in tqdm(range(seq_len)):
            z = torch.randn((batch_size, NPP.cvae.noise_dim))
            xhats = NPP.cvae.decode(z, h.squeeze(0)).unsqueeze(1)               # [batch_size, 1, data_dim]
            points.append(xhats.squeeze(1))
            out, (h, c) = NPP.lstm(torch.permute(xhats, (1, 0, 2)), (h, c)) 
        
    # collection of events
    points = torch.stack(points, dim=1)
    points[:, :, 0] = torch.cumsum(points[:, :, 0], dim=1)

    return points.cpu().numpy()                # [batch_size, seq_len, data_dim]