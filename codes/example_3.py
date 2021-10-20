# -*- coding: utf-8 -*-
"""
Implementation of the Example 3 from section 4 of

Echard et. al. (2011) - AK-MCS: An active learning reliability method combining 
Kriging and Monte Carlo Simulation. Structural Safety. 33. p. 145-154
https://doi.org/10.1016/j.strusafe.2011.01.002

Authors:
JCLR - Juan Camilo Lugo Rojas      jclugor@unal.edu.co
DAAM - Diego Andrés Alvarez Marín  daalvarez@unal.edu.co

DATE          WHO   WHAT
Oct 13, 2021  JCLR  Algorithm
Oct 13, 2021  DAAM  Comments and readability
"""
# %% modules import
import numpy as np
from functions import ak_mcs, plot_conv_pf, plot_learn_criterion
import learning_functions as lf

# %% performance function

def G(X):
    m, c1, c2, r, F1, t1 = X.T
    w0 = np.sqrt((c1 + c2)/m)
    return 3*r - np.abs(2*F1/(m*w0**2)*np.sin(w0*t1/2))

# %% Generation of a Monte Carlo population in the design space
n_MC = 70_000      # number of points in the MC population

# the considered variables follow the normal distributions:
random_variables = [ lambda n: np.random.normal(  1, 0.05, n),  # m   
                     lambda n: np.random.normal(  1,  0.1, n),  # c1
                     lambda n: np.random.normal(0.1, 0.01, n),  # c2
                     lambda n: np.random.normal(0.5, 0.05, n),  # r 
                     lambda n: np.random.normal(  1,  0.2, n),  # F1
                     lambda n: np.random.normal(  1,  0.2, n) ] # t1

# %% computation of the AK-MCS algorithm
N1 = 12    # size of the initial DoE

# learning functions to be tested
lfuncs = [ lf.learning_fun_U,
           lf.learning_fun_EFF,
           lf.learning_fun_H    ]

results = {}
for lfun in lfuncs:
    np.random.seed(seed=1234)
    lf_results = zip(['S', 'idx_DoE', 'y_DoE', 'list_pf_hat', 'list_gp', 'list_lf_xstar'],
                     ak_mcs(random_variables, n_MC, G, lfun, N1=N1))
    # S, idx_DoE, y_DoE, list_pf_hat, list_gp, list_lf_xstar = \
    #                                        ak_mcs(random_variables, n_MC, G, lf, N1=N1)
    results[lfun] = {}
    for key, value in lf_results:
        results[lfun][key] = value

# %% solution of MCS
S           = results[lf.learning_fun_U]['S']
fail_points = G(S) <= 0
pf_MCS      = fail_points.mean()
CoV         = np.sqrt((1-pf_MCS)/(pf_MCS*n_MC))
    
# %% plot results
for lfun in lfuncs:
    plot_conv_pf(N1, results[lfun]['list_pf_hat'], pf_MCS, lfun.name)

    # %% plot minimum values of learning function 
    plot_learn_criterion(N1, results[lfun]['list_lf_xstar'], lfun.name, lfun.threshold)

