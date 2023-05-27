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

from KDE_CEG import NeuralPP


def train(config, model_name, save_path):

    # Create folder
    if os.path.exists(save_path): 
        raise ValueError("Duplicated folder!")
    else:
        os.makedirs(save_path)

    # Load data and separate them as training/testing set
    data         = torch.Tensor(np.load(config['data_path']))
    if config['data_dim'] == 1:
        data         = data.unsqueeze(-1)
    train_loader = DataLoader(torch.utils.data.TensorDataset(data[:config['train_size']]), 
                              shuffle=True, batch_size=config['batch_size'], drop_last=True)
    test_loader  = DataLoader(torch.utils.data.TensorDataset(data[config['train_size']:]), 
                              shuffle=True, batch_size=config['batch_size'], drop_last=True)
    
    # set sequence length
    seq_len           = data.shape[1]
    config['seq_len'] = seq_len

    # set adaptive bandwidth
    kde_bdw_base = torch.Tensor(config['kde_bdw_base']).to(config['device'])
    kde_bdw_decay = config['kde_bdw_decay']

    # initialize model
    model        = NeuralPP(config)
    if config['warm_start']:
        print("Warm start. Loading model...")
        model.load_state_dict(torch.load(config['warm_path']))
        # model.kde_bdw.data = torch.Tensor([0.5])[0]
    model.to(config['device'])

    # initialize optimizer
    optimizer    = Adam(model.parameters(), lr=config["lr"])
    
    # iteration
    avg_nll, avg_test_nll = [], []
    n_events_train, n_events_test = 0, 0
    for epoch in range(config['epochs']):
        try:
            # training
            model.train()
            for batch in train_loader:
                # init optimizer 
                optimizer.zero_grad()     
                # negative log-likelihood
                X         = batch[0].to(config["device"])
                ll, n_events, _, _  = model.log_liklihood(X)
                nll       = - ll
                avg_nll.append(nll.item())
                n_events_train += n_events
                # optimize
                nll.backward()             
                optimizer.step()

                for para in model.parameters():
                    if torch.isnan(para).any(): raise ValueError("NaN encountered!!!")

            # testing
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    test_X        = batch[0].to(config["device"])
                    test_ll, n_events, _, _ = model.log_liklihood(test_X)
                    test_nll      = - test_ll
                    avg_test_nll.append(test_nll.item())
                    n_events_test += n_events

            if (epoch + 1) % config['prt_evry'] == 0:
                print("Epochs:%d" % epoch)
                print("training nll:%.3e" % (sum(avg_nll) / n_events_train))
                print("testing nll:%.3e" % (sum(avg_test_nll) / n_events_test))
                avg_nll, avg_test_nll = [], []

            if (epoch + 1) % config['log_iter'] == 0:
                print("Saving model...")
                model.to("cpu")
                torch.save(model.state_dict(), save_path + "/%s-%d.pth" % (model_name, epoch))
                model.to(config['device'])

            n_events_train, n_events_test = 0, 0
            # adaptive bandwidth
            model.kde_bdw.data = (model.kde_bdw.data - kde_bdw_base) * kde_bdw_decay + kde_bdw_base
            print("Model bandwidth: ", model.kde_bdw.data.cpu().numpy())
            
        except KeyboardInterrupt:
            break
    
    model.to("cpu")
    return model
    
seed = 100
torch.manual_seed(seed)

data_load = "data-1d-Exp-train-mu0.5-beta1.0-size120-n1000"
data_path = "data/%s.npy" % data_load
train_size = 900

model_name = "%s_npp" % (data_load + "-RefKDE-dym-bdw-z0.2")
save_path = "saved_models/%s" % model_name
warm_path = "saved_models/%s_stage1/%s-%d.pth" % (model_name, model_name, 199)

train_config = {
    'data': 'self-correcting_hawkes',
    'data_path': data_path,
    'train_size': train_size,
    'data_dim': 1,
    'hid_dim': 32,
    'mlp_layer': 2,
    'kde_bdw':[.2],
    'kde_bdw_base':[0.05],
    'kde_bdw_decay':0.9,
    'n_samples': 1000,
    'noise_dim': 10,
    'mlp_dim': 32,
    'batch_size':32,
    'epochs': 200,
    'lr': 1e-3,
    'dropout': 0.1,
    'prt_evry':1,
    'early_stop': False,
    'alpha': 0.05,
    'log_mode': False,
    'log_iter': 5,
    'warm_start': False,
    'warm_path': warm_path,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

model = train(train_config, model_name, save_path)