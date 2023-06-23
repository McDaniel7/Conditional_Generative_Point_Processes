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

from VAE_CEG import NeuralPP

# Training

def train(config, model_name, save_path):

    # Create folder
    if os.path.exists(save_path): 
        raise ValueError("Duplicated folder!")
    else:
        os.makedirs(save_path)

    # Load data and separate them as training/testing set
    data         = torch.Tensor(np.load(config['data_path']))
    if config["data_dim"] == 1: data = data.unsqueeze(-1)
    data         = data[:, :150, :]
    train_loader = DataLoader(torch.utils.data.TensorDataset(data[:config['train_size']]), 
                              shuffle=True, batch_size=config['batch_size'], drop_last=True)
    test_loader  = DataLoader(torch.utils.data.TensorDataset(data[config['train_size']:]), 
                              shuffle=True, batch_size=config['batch_size'], drop_last=True)
    
    # set sequence length
    seq_len           = data.shape[1]
    config['seq_len'] = seq_len

    # initialize model
    model        = NeuralPP(config)
    if config['warm_start']:
        print("Warm start. Loading model...")
        model.load_state_dict(torch.load(config['warm_path']))
    model.to(config['device'])

    # initialize optimizer
    optimizer    = Adam(model.parameters(), lr=config["lr"])
    
    # iteration
    avg_loss, avg_test_loss = [], []
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
                loss, n_events, _  = model.log_liklihood(X)
                avg_loss.append(loss.item())
                n_events_train += n_events
                # optimize
                loss.backward()             
                optimizer.step()

                for para in model.parameters():
                    if torch.isnan(para).any(): raise ValueError("NaN encountered!!!")

            # testing
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    test_X        = batch[0].to(config["device"])
                    test_loss, n_events, _ = model.log_liklihood(test_X)
                    avg_test_loss.append(test_loss.item())
                    n_events_test += n_events

            if (epoch + 1) % config['prt_evry'] == 0:
                print("Epochs:%d" % epoch)
                print("training loss:%.3e" % np.mean(avg_loss))
                print("testing loss:%.3e" % np.mean(avg_test_loss))
                avg_loss, avg_test_loss = [], []

            if (epoch + 1) % config['log_iter'] == 0:
                print("Saving model...")
                model.to("cpu")
                torch.save(model.state_dict(), save_path + "/%s-%d.pth" % (model_name, epoch))
                model.to(config['device'])

            n_events_train, n_events_test = 0, 0
            
        except KeyboardInterrupt:
            break
    
    model.to("cpu")
    return model
    
seed = 100
torch.manual_seed(seed)

data_load = "data-1d-SelfCorrecting-train-mu1.0-alpha1.0-size150-n1000"
data_path = "data/%s.npy" % data_load
train_size = 900
mlp_layer = 1
mlp_dim   = 2048
noise_dim = 32
hid_dim   = 32

model_name = "%s_npp" % (data_load + "-CVAE-layer%d-mlp%d-embed%d-hid%d_xavier" % (mlp_layer, mlp_dim, noise_dim, hid_dim))
save_path = "saved_models/%s" % model_name
warm_path = "saved_models/%s_stage1/%s-%d.pth" % (model_name, model_name, 99)

train_config = {
    'data': 'self-correcting',
    'data_path': data_path,
    'train_size': train_size,
    'data_dim': 1,
    'hid_dim': hid_dim,
    'noise_dim': noise_dim,
    'mlp_dim': mlp_dim,
    'mlp_layer': mlp_layer,
    'batch_size':100,
    'epochs': 200,
    'lr': 1e-3,
    'prt_evry':1,
    'early_stop': False,
    'alpha': 0.05,
    'log_mode': False,
    'log_iter': 5,
    'warm_start': True,
    'warm_path': warm_path,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

model = train(train_config, model_name, save_path)