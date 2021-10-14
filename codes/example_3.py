# -*- coding: utf-8 -*-
"""
Implementation of the Example 3 from section 4 of "AK-MCS: An active learning
reliability method combining Kriging and Monte Carlo Simulation" by B. Echard
et al (DOI: 10.1016/j.strusafe.2011.01.002)

@author: jclugor
"""
# %% modules import
import numpy as np
from functions import ak_mcs, plot_conv, plot_lim_vals
import learning_functions_DIEGO as lf

# %% performance functio

def G(X):
    c1, c2, m, r, t, F = X.T
    w = np.sqrt((c1 + c2)/m)
    return 3*r - np.abs(2*F/(m*w**2)*np.sin(w*t/2))

# %% Generation of a Monte Carlo population in the design space
n_MC = int(7e4)

# the considered variables follow the normal distributions:
normal_1 = lambda n: np.random.normal(1,   0.1,  n)
normal_2 = lambda n: np.random.normal(0.1, 0.01, n)
normal_3 = lambda n: np.random.normal(1,   0.05, n)
normal_4 = lambda n: np.random.normal(0.5, 0.05, n)
normal_5 = lambda n: np.random.normal(1,   0.2,  n)

var_dists = [normal_1,
             normal_2,
             normal_3,
             normal_4,
             normal_5,
             normal_5]


# =============================================================================
# # %% solution by MCS
# fail_points = G(X) <= 0
# pf_mc = fail_points.mean()
# cov = np.sqrt((1-pf_mc)/(pf_mc*n_MC))
# methods = {'Monte Carlo': [n_MC, pf_mc,cov]}
# 
# =============================================================================
# %% computation of the AK-MCS algorithm
# learning functions to be tested
               # fun   threshold 
lfuncs = {'U':   [lf.learning_fun_U,   2],
          'EFF': [lf.learning_fun_EFF, 0.001],
          'H':   [lf.learning_fun_H,   0.5]}

for fun in lfuncs:
    results, plot_DoE, plot_y = ak_mcs(var_dists, n_MC, G, lfuncs[fun][0], k=11)    
    # error = (pf_mc - results[-1,2])/pf_mc*100
    # store results
    # methods[f'AK-MCS+{fun}'] = [results[-1,0], results[-1,2], error]    
    
# =============================================================================
#     # %% plot minimum values of learning function 
#     plot_lim_vals(results, lfuncs[fun][1], fun, 'ex_3', save=True)
#     
#     # %% plot convergence of pf
#     plot_conv(results, pf_mc, fun, ex_name='ex_3', save=True)
# =============================================================================
 #%%
# print results
for method in methods:
    print(f'{method}:')
    print(methods[method])
