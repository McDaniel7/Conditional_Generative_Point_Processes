#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TPPG: Temporal Point Process Generator

Dependencies:
- Python 3.6.7
"""

import sys
import utils
import arrow
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt


from tppg import ExpKernel, RayleighKernel, HawkesLam, SelfCorrectingLam, TemporalPointProcess

############ 1d self-exciting Hawkes #############
# parameters initialization
mu     = .2
T      = [0., 100.]
beta   = .4
kernel = ExpKernel(beta=beta)
lam    = HawkesLam(mu, kernel, maximum=1.0e2)
pp     = TemporalPointProcess(lam)

seed = 200
np.random.seed(seed)
# generate points
n_seqs = 1000
data, sizes = pp.generate(T=T, batch_size=n_seqs, verbose=False)

############ 1d self-correcting Hawkes #############
# parameters initialization
mu     = .5
T      = [0., 200.]
alpha  = .8
lam    = SelfCorrectingLam(mu, alpha, maximum=2e+1)
pp     = TemporalPointProcess(lam)

seed = 100
np.random.seed(seed)
# generate points
n_seqs = 1000
data, sizes = pp.generate(T=T, batch_size=n_seqs, verbose=False)