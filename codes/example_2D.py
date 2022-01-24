# -*- coding: utf-8 -*-
"""
Implementation of the Example from section 4.1 of "AK-MCS: An active learning
reliability method combining Kriging and Monte Carlo Simulation" by B. Echard
et al (DOI: 10.1016/j.strusafe.2011.01.002)

@author: jclugor
"""
# %% modules import
import numpy as np
from functions import ak_mcs, mc_population, plot_conv_pf, plot_learn_criterion, plot_mc, plot_iter
import learning_functions as lf

# %% performance function
# a series system of four branches
# it is tested with k=6 and k=7

def G(S, k=7):
    S = np.atleast_2d(S).copy()
    g = np.empty((len(S), 4))
    g[:,0] = 3 + 0.1*(S[:,0] - S[:,1])**2 - (S[:,0] + S[:,1])/np.sqrt(2)
    g[:,1] = 3 + 0.1*(S[:,0] - S[:,1])**2 + (S[:,0] + S[:,1])/np.sqrt(2)
    g[:,2] = S[:,0] - S[:,1] + k/np.sqrt(2)
    g[:,3] = S[:,1] - S[:,0] + k/np.sqrt(2)
    return g.min(axis=1)

# %% Generation of a Monte Carlo population in the design space
n_MC = 1_000_000    # number of points in the design space

# the two considered variables follow a standard normal distribution
random_variables = [lambda n: np.random.normal(size= n)] * 2

# %% solution by MCS
seed = 1234     # random seed to generate the MC population
np.random.seed(seed=seed)
S = mc_population(random_variables, n_MC)
fail_points = G(S) <= 0
pf_MCS      = fail_points.mean()
CoV         = np.sqrt((1-pf_MCS)/(pf_MCS*n_MC))

plot_mc(S, fail_points, pf_MCS)
# %% computation of the AK-MCS algorithm
N1 = 12    # size of the initial DoE

# learning functions to be tested
lfuncs = [lf.learning_fun_U,
          lf.learning_fun_EFF,
          lf.learning_fun_H    ]

for lfun in lfuncs:
    np.random.seed(seed=seed)
    idx_DoE, y_DoE, list_pf_hat, list_gp, list_lf_xstar = \
                                  ak_mcs(random_variables, n_MC, G, lfun, N1)

# %% plot results
    plot_conv_pf(N1, list_pf_hat, pf_MCS, lfun.name)

    # %% plot minimum values of learning function 
    plot_learn_criterion(N1, list_lf_xstar, lfun)

    # %% plot predicted failure through the iterations
    plot_iter(S, N1, idx_DoE, y_DoE, list_gp)
    
