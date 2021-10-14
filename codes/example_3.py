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
from functions import ak_mcs, plot_conv_pf, plot_lim_vals
import learning_functions as lf

# %% performance function

def G(X):
    m, c1, c2, r, F1, t1 = X.T
    w0 = np.sqrt((c1 + c2)/m)
    return 3*r - np.abs(2*F1/(m*w0**2)*np.sin(w0*t1/2))

# %% Generation of a Monte Carlo population in the design space
n_MC = 70_000

# the considered variables follow the normal distributions:
random_variables = [ lambda n: np.random.normal(  1, 0.05, n),  # m   
                     lambda n: np.random.normal(  1,  0.1, n),  # c1
                     lambda n: np.random.normal(0.1, 0.01, n),  # c2
                     lambda n: np.random.normal(0.5, 0.05, n),  # r 
                     lambda n: np.random.normal(  1,  0.2, n),  # F1
                     lambda n: np.random.normal(  1,  0.2, n) ] # t1

# %% computation of the AK-MCS algorithm
N1 = 12

# learning functions to be tested
               # fun   threshold 
lfuncs = [ lf.learning_fun_U,
           lf.learning_fun_EFF,
           lf.learning_fun_H    ]

for lf in lfuncs:
    np.random.seed(seed=1234)
    S, idx_DoE, y_DoE, list_pf_hat, list_gp, list_lf_xstar = \
                                           ak_mcs(random_variables, n_MC, G, lf, N1=N1)

    y = G(S)
    pf_MCS = (y <= 0).mean()

    plot_conv_pf(N1, list_pf_hat, pf_MCS, lf.name)


    # %% plot minimum values of learning function 
    plot_lim_vals(N1, list_lf_xstar, lf.name, lf.threshold)
 #%%
# print results
'''
for method in methods:
    print(f'{method}:')
    print(methods[method])
'''    


# =============================================================================
# # %% solution by MCS
# fail_points = G(X) <= 0
# pf_mc = fail_points.mean()
# cov = np.sqrt((1-pf_mc)/(pf_mc*n_MC))
# methods = {'Monte Carlo': [n_MC, pf_mc,cov]}
# 
# =============================================================================