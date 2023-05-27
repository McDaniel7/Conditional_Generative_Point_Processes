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


############### 1d data evaluation ##############

def evaluate(data, config, T, ngrid):
    # Load data and separate them as training/testing set
    eval_loader  = DataLoader(torch.utils.data.TensorDataset(data), 
                              shuffle=False, batch_size=config['batch_size'], drop_last=True)
    
    # set sequence length
    seq_len           = data.shape[1]
    config['seq_len'] = seq_len
    
    # load model
    model = NeuralPP(config)
    model.load_state_dict(torch.load(config["saved_path"]))
    # model.kde_bdw.data = torch.tensor([0.5])[0]
    print("Model KDE bandwidth: %.3f" % model.kde_bdw.cpu().numpy())
    # create a grid over the entire space, where each point of the grid will be evaluated. 
    grid         = torch.Tensor(np.linspace(T[0], T[1], ngrid))
    grid         = torch.tile(grid.unsqueeze(0).unsqueeze(-1), (config["batch_size"], 1, 1))
                                                        # [batch_size, ngrid, 1]
    
    fss = []
    for batch in eval_loader:
        with torch.no_grad():
            model.eval()
            fs = model.lambda_(grid, batch[0], n_sample=500)
            fss.append(fs.detach().numpy())

    fss = np.concatenate(fss, axis=0)
    
    return fss


def plt_lam(points, lam, est_fs, est_Fs=None, est_lamvals=None, T=[0., 1.], ngrid=100):
    """
    - lam: true model, 'HawkesLam' (or 'SelfCorrectingLam') object
    """
    points  = points[points > 0]
    eval_points = points[(points <= T[1]) * (points >= T[0])]
    ts      = np.linspace(T[0], T[1], ngrid)
    unit_vol = (T[1]-T[0]) / (ngrid-1)
    zros    = np.zeros((len(eval_points)))
    # print(points.shape, zros.shape)
    lamvals = []
    evals   = []
    fvals   = []
    Fvals   = []
    
    if est_Fs is None:
        est_Fs = []
        for i, t in enumerate(ts):
            his_t  = points[(points < t) * (points > 0)]
            last_t = his_t[-1] if len(his_t) > 0 else 0.
            last_ts_idx = (ts < last_t).sum()
            est_fint = np.array(est_fs[last_ts_idx:(i+1)])
            tint = ts[last_ts_idx:(i+1)]
            est_F_integral = (np.array([tint[0] - last_t] + [unit_vol] * (len(tint) - 1)) * est_fint).sum()
            est_Fs.append(est_F_integral)
    
    if est_lamvals is None:
        est_lamvals = np.array(est_fs) / (1 - np.array(est_Fs))

    for i, t in enumerate(ts):
        his_t  = points[(points < t) * (points > 0)]
        lamval = lam.value(t, his_t)
        lamvals.append(lamval)
        last_t = his_t[-1] if len(his_t) > 0 else 0.
        last_ts_idx = (ts < last_t).sum()
        lamint = np.array(lamvals[last_ts_idx:])
        tint = ts[last_ts_idx:(i+1)]
        integral = (np.array([tint[0] - last_t] + [unit_vol] * (len(tint) - 1)) * lamint).sum()
        fvals.append(lamval * np.exp(-integral))
        fint = np.array(fvals[last_ts_idx:])
        F_integral = (np.array([tint[0] - last_t] + [unit_vol] * (len(tint) - 1)) * fint).sum()
        Fvals.append(F_integral)
    
    # true lambda computed by true fs and Fs
    lamvals = np.array(fvals) / (1 - np.array(Fvals))

    for t in eval_points:
        his_t  = points[(points < t) * (points > 0)]
        lamval = lam.value(t, his_t)
        evals.append(lamval)

    fig = plt.figure(figsize=(18, 4))
    
    ax1 = fig.add_subplot(131)
    ax1.plot(ts, lamvals, linestyle="--", color="grey", label="r$\lambda(t)$")
    ax1.plot(ts, est_lamvals, linestyle="-", color="red", label="r$\hat{\lambda}(t)$")
    ax1.scatter(eval_points, zros - 0.1, c="black")
    ax1.legend(fontsize=15)
    # ax1.set_ylim(0, max(lamvals) * 1.1)

    ax2 = fig.add_subplot(132)
    ax2.plot(ts, fvals, linestyle="--", color="grey", label="r$f(t)$")
    ax2.plot(ts, est_fs, linestyle="-", color="blue", label="r$\hat{f}(t)$")
    ax2.scatter(eval_points, zros - 0.1, c="black")
    ax2.legend(fontsize=15)

    ax3 = fig.add_subplot(133)
    ax3.plot(ts, Fvals, linestyle="--", color="grey", label="r$F(t)$")
    ax3.plot(ts, est_Fs, linestyle="-", color="green", label="r$\hat{F}(t)$")
    ax3.scatter(eval_points, zros - 0.1, c="black")
    ax3.legend(fontsize=15)

    plt.show()

    return lamvals, fvals, Fvals, est_lamvals, est_fs, est_Fs



################################ 3d data evaluation ###################################
############### only implemented for model by non-parametric learning #################

def evaluate_3d(points, eval_ts, config, T=[0., 1.], S=[[0., 1.], [0., 1.]],
                int_ngrid=50, plot_ngrid=100):
    """
    - points: tensor, [seq_len, data_dim]
    - eval_ts: scalar
    """
    points   = points[(points[:, 0] < eval_ts) * (points[:, 0] > 0)]
    last_t   = points[-1, 0] if len(points) > 0 else 0.
    last_s   = points[-1, 1:] if len(points) > 0 else torch.zeros(config["data_dim"]-1)
    points   = points.unsqueeze(0)
    # set sequence length
    seq_len           = points.shape[1]
    config['seq_len'] = seq_len
    
    # load model
    model = NeuralPP(config)
    model.load_state_dict(torch.load(config["saved_path"]))

    # create a grid over the location space, where each point of the grid will be evaluated.
    ts      = np.linspace(last_t, eval_ts, int_ngrid)
    ss      = [np.linspace(S_k[0], S_k[1], plot_ngrid) for S_k in S]
    ss      = torch.Tensor(list(product(*ss)))

    with torch.no_grad():
        h0     = torch.zeros([1, model.batch_size, model.hid_dim], device=points.device)
        c0     = torch.zeros([1, model.batch_size, model.hid_dim], device=points.device)
        dX     = model._delta_X(points)                 # [1, seq_len, data_dim]
        out, (_, _) = model.lstm(torch.permute(dX, (1, 0, 2)), (h0, c0))
                                                     # out=[seq_len, 1, hid_dim]
        hs     = out[-1, :, :]                                    # [1, hid_dim]
        xhats  = model.fnet(hs)                    # [1, nsamples, data_dim]

        # generate xhats for KDE
        def calc_sum_fs(t):
            input = torch.concat((torch.ones_like(ss[:, [0]]) * (t-last_t), ss), dim=-1)
            fs     = model._fKDE(input, xhats, h=model.kde_bdw)
            return fs.sum()

        sum_fs = [calc_sum_fs(t) for t in tqdm(ts[1:])]
        input = torch.concat((torch.ones_like(ss[:, [0]]) * (eval_ts-last_t), ss), dim=-1)
        fs     = model._fKDE(input, xhats, h=model.kde_bdw)
        
        Fs = np.sum(sum_fs) * (eval_ts - last_t) / (int_ngrid - 1) * \
                np.prod([S_k[1] - S_k[0] for S_k in S]) / (plot_ngrid - 1)**2

    return fs, Fs

def plt_lam_3d(plot_ts, points, lam, est_fs_at_ts, est_Fs_at_ts, est_lamvals_at_ts=None,
               T=[0., 1.], S=[[0., 1.], [0., 1.]],
               int_ngrid=50, plot_ngrid=100):
    """
    Plot true and learned intensity and probability density function
    """
    ss      = [np.linspace(S_k[0], S_k[1], plot_ngrid) for S_k in S]
    ss      = np.array(list(product(*ss)))
    loc_unit_vol  = np.prod([S_k[1] - S_k[0] for S_k in S]) / (plot_ngrid - 1)**2

    def lamval_at_t(t):
        his_p   = points[(points[:, 0] < t) * (points[:, 0] > 0)]
        his_t   = his_p[:, 0]
        his_s   = his_p[:, 1:]
        lams    = [lam.value(plot_ts, his_t, s, his_s) for s in ss]
        return lams

    lamvals       = []
    lamvals_at_ts = []
    fvals         = []
    fvals_at_ts   = []

    # true intensity
    lamvals_at_ts = lamval_at_t(plot_ts)
    lamvals_at_ts = np.array(lamvals_at_ts)

    # true density
    his_p   = points[(points[:, 0] < plot_ts) * (points[:, 0] > 0)]
    his_t   = his_p[:, 0]
    last_t  = his_t[-1] if len(his_t) > 0 else 0.
    ts      = np.linspace(last_t, plot_ts, int_ngrid)
    time_unit_vol = (plot_ts - last_t) / (int_ngrid - 1)
    unit_vol = time_unit_vol * loc_unit_vol
    # for t in ts[1:]:
    #     for s in ss:
    #         lamval = lam.value(t, his_t, s, his_s)
    #         lamvals.append(lamval)
    lamvals = [lamval_at_t(t) for t in ts[1:]]
    integral     = np.sum(lamvals) * unit_vol
    fvals_at_ts  = lamvals_at_ts * np.exp(-integral)

    # true F
    Fvals_at_ts = 1 - np.exp(-integral)
    
    # estimated intensitys
    if est_lamvals_at_ts is None:
        est_lamvals_at_ts = np.array(est_fs_at_ts) / (1 - est_Fs_at_ts)

    # true lambda computed by true fs and Fs
    lamvals = fvals_at_ts / (1 - Fvals_at_ts)

    fig = plt.figure(figsize=(8, 7))
    
    ax1 = fig.add_subplot(2, 2, 1)
    im = ax1.imshow(lamvals_at_ts.reshape(plot_ngrid,plot_ngrid).T)
    ax1.set_xticks([0, plot_ngrid-1])
    ax1.set_xticklabels([S[0][0], S[0][1]])
    ax1.set_yticks([0, plot_ngrid-1])
    ax1.set_yticklabels([S[1][0], S[1][1]])
    fig.colorbar(im, ax=ax1, shrink=0.8)
    ax1.set_title(r"True $\lambda(%.1f, \cdot)$" % plot_ts, fontsize=18)

    ax2 = fig.add_subplot(2, 2, 2)
    im = ax2.imshow(fvals_at_ts.reshape(plot_ngrid,plot_ngrid).T, cmap="magma")
    ax2.set_xticks([0, plot_ngrid-1])
    ax2.set_xticklabels([S[0][0], S[0][1]])
    ax2.set_yticks([0, plot_ngrid-1])
    ax2.set_yticklabels([S[1][0], S[1][1]])
    fig.colorbar(im, ax=ax2, shrink=0.8)
    ax2.set_title(r"True $f(%.1f, \cdot)$" % plot_ts, fontsize=18)

    ax3 = fig.add_subplot(2, 2, 3)
    im = ax3.imshow(est_lamvals_at_ts.reshape(plot_ngrid,plot_ngrid).T)
    ax3.set_xticks([0, plot_ngrid-1])
    ax3.set_xticklabels([S[0][0], S[0][1]])
    ax3.set_yticks([0, plot_ngrid-1])
    ax3.set_yticklabels([S[1][0], S[1][1]])
    fig.colorbar(im, ax=ax3, shrink=0.8)
    ax3.set_title(r"Learned $\lambda(%.1f, \cdot)$" % plot_ts, fontsize=18)

    ax4 = fig.add_subplot(2, 2, 4)
    im = ax4.imshow(est_fs_at_ts.reshape(plot_ngrid,plot_ngrid).T, cmap="magma")
    ax4.set_xticks([0, plot_ngrid-1])
    ax4.set_xticklabels([S[0][0], S[0][1]])
    ax4.set_yticks([0, plot_ngrid-1])
    ax4.set_yticklabels([S[1][0], S[1][1]])
    fig.colorbar(im, ax=ax4, shrink=0.8)
    ax4.set_title(r"Learned $f(%.1f, \cdot)$" % plot_ts, fontsize=18)

    plt.show()

    return lamvals_at_ts, fvals_at_ts, est_lamvals_at_ts, est_fs_at_ts


def plot_NPP_spatial_density(NPP, points, S, plot_tlag, t_slots, grid_size, interval):
    """
    Plot spatial intensity as the time goes by. The generated points can be also
    plotted on the same 2D space optionally.
    """
    assert len(S) == 3, '%d is an invalid dimension of the space.' % len(S)
    # remove zero points
    points = points[points[:, 0] > 0]
    # split points into sequence of time and space.
    seq_t, seq_s = points[:, 0], points[:, 1:]
    # define the span for each subspace
    t_span  = np.linspace(S[0][0], S[0][1], t_slots+1)[1:]
    ss      = [np.linspace(S_k[0], S_k[1], grid_size+1)[:-1] for S_k in S[1:]]
    ss      = torch.Tensor(list(product(*ss)))
    # set sequence length
    points = points.unsqueeze(0)
    seq_len = points.shape[1]

    # function for yielding the heatmap over the entire region at a given time
    with torch.no_grad():
        h0     = torch.zeros([1, NPP.batch_size, NPP.hid_dim], device=points.device)
        c0     = torch.zeros([1, NPP.batch_size, NPP.hid_dim], device=points.device)
        dX     = NPP._delta_X(points)                       # [1, seq_len, data_dim]
        out, (_, _) = NPP.lstm(torch.permute(dX, (1, 0, 2)), (h0, c0))
                                                        # out=[seq_len, 1, hid_dim]
        hs     = torch.concat([h0, out], dim=0)[:, 0, :]      # [seq_len+1, hid_dim]
        xhats  = NPP.fnet(hs)                      # [seq_len+1, nsamples, data_dim]

    def heatmap_fs(t):
        with torch.no_grad():
            sub_seq_t = seq_t[seq_t < t]
            idx    = len(sub_seq_t)
            last_t = sub_seq_t[-1] if idx > 0 else 0
            input = torch.concat((torch.ones_like(ss[:, [0]]) * (t-last_t), ss), dim=-1)
            fs     = NPP._fKDE(input, xhats[[idx]], h=NPP.kde_bdw)
        return fs.cpu().numpy().reshape(grid_size, grid_size).T[::-1]

    def plot_events(t):
        sub_seq_t = seq_t[(seq_t >= t - plot_tlag) * (seq_t < t)]
        sub_seq_s = seq_s[(seq_t >= t - plot_tlag) * (seq_t < t)]
        return sub_seq_t, sub_seq_s

    # calculate density 
    # prepare the heatmap data in advance
    print('[%s] preparing the dataset %d × (%d, %d) for plotting.' %
        (arrow.now(), t_slots, grid_size, grid_size), file=sys.stderr)
    fvals = [heatmap_fs(t) for t in tqdm(t_span)]
    fvals = np.array(fvals)
    plot_seq = [plot_events(t) for t in tqdm(t_span)]

    # initiate the figure and plot
    fig = plt.figure()
    # set the image with largest total intensity as the intial plot for automatically setting color range.
    # im  = plt.imshow(data[data.sum(axis=-1).sum(axis=-1).argmax()], cmap='hot', animated=True) 
    im  = plt.imshow(fvals[-1], cmap='Oranges', animated=True)
    scat = plt.scatter(0.5, 0.5, s=50, c="blue", marker="x", alpha=0)
    # function for updating the image of each frame
    def animate(i):
        # print(t_span[i])
        im.set_data(fvals[i])
        plot_t, plot_s = plot_seq[i]
        if len(plot_t) > 0:
            # plot_s[:, 0] = plot_s[:, 0] * (grid_size - 1)
            # # plot_s[:, 1] = (1 - plot_s[:, 1]) * (grid_size - 1)
            # plot_s[:, 1] = plot_s[:, 1] * (grid_size - 1)
            plot_s = plot_s * (grid_size - 1)
            plot_s[:, 1] = grid_size - 1 - plot_s[:, 1]
            scat.set_offsets(plot_s)
            scat.set_sizes([100] * len(plot_t))
            scat.set_alpha(1 - (t_span[i] - plot_t) / plot_tlag)
        else:
            scat.set_offsets(plot_s)
        return im, scat
    # function for initiating the first image of the animation
    def init():
        im.set_data(fvals[0])
        return im, scat
    # animation
    print('[%s] start animation.' % arrow.now(), file=sys.stderr)
    anim = animation.FuncAnimation(fig, animate,
        init_func=init, frames=t_slots, interval=interval, blit=True)
    
    return anim