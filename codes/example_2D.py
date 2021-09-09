# -*- coding: utf-8 -*-
"""
Implementation of the Example from section 4.1 of "AK-MCS: An active learning
reliability method combining Kriging and Monte Carlo Simulation" by B. Echard
et al (DOI: 10.1016/j.strusafe.2011.01.002)

@author: jclugor
"""
# %% modules import
import numpy as np
from functions import ak_mcs, plot_conv, plot_lim_vals, plot_mc, plot_iter
import learning_functions as lf

# %% performance function
# a series system of four branches
# it is tested with k=6 and k=7

def G(X, k=7):
    g = np.empty((len(X), 4))
    g[:,0] = 3 + 0.1*(X[:,0] - X[:,1])**2 - (X[:,0] + X[:,1])/np.sqrt(2)
    g[:,1] = 3 + 0.1*(X[:,0] - X[:,1])**2 + (X[:,0] + X[:,1])/np.sqrt(2)
    g[:,2] = X[:,0] - X[:,1] + k/np.sqrt(2)
    g[:,3] = X[:,1] - X[:,0] + k/np.sqrt(2)
    return g.min(axis=1)

# %% Generation of a Monte Carlo population in the design space
N = int(1e6)
# the two considered variables follow a standard normal distribution
X = np.random.normal(size=(N,2))

# %% solution by MCS
fail_points = G(X) <= 0
pf_mc = fail_points.mean()
cov = np.sqrt((1-pf_mc)/(pf_mc*N))
methods = {'Monte Carlo': [N, pf_mc,cov]}

plot_mc(X, fail_points, pf_mc, ex_name='ex_2D_k7', save=True)
# %% computation of the AK-MCS algorithm
# learning functions to be tested
               # fun   threshold 
lfuncs = {'U':   [lf.U,   2],
          'EFF': [lf.EFF, 0.001],
          'H':   [lf.H,   0.5]}

for fun in lfuncs:
    results, plot_DoE, plot_y = ak_mcs(X, G, lfuncs[fun][0], k=11)    
    error = (pf_mc - results[-1,2])/pf_mc*100
    # store results
    methods[f'AK-MCS+{fun}'] = [results[-1,0], results[-1,2], error]    
    # %% plot predicted failure through the iterations
    plot_iter(X, plot_y, plot_DoE, ex_name=f'ex_2D_k7_{fun}', save=True)
    
    # %% plot minima values of learning function 
    plot_lim_vals(results, lfuncs[fun][1], fun, 'ex_2D_k7', save=True)
    
    # %% plot convergence of pf
    plot_conv(results, pf_mc, fun, ex_name='ex_2D_k7', save=True)

# print results
for method in methods:
    print(f'{method}:')
    print(methods[method])
