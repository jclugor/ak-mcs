# -*- coding: utf-8 -*-
"""
Example to illustrate how does the method select the points to be added to the DoE

@author: jclugor
"""
# %% modules import
import numpy as np
from functions import ak_mcs, plot_1d
import learning_functions as lf

# %% performance function
def G(X):
    return np.sin(X).flatten()

# %% Generation of a Monte Carlo population in the design space
N = int(1e4)
X = np.random.normal(0, 2, size=(N,1))

# %% computation of the AK-MCS algorithm
results, plot_DoE, plot_y, plot_std = ak_mcs(X, G, lf.U, 5, k=1, std=True)

# %% solution by MCS
fail_points = G(X) <= 0
pf_mc = fail_points.mean()
cov = np.sqrt((1-pf_mc)/(pf_mc*N))

# %% results
plot_1d(X, G, plot_y, plot_DoE, plot_std, save=True, ex_name='exa_1d')
error = (pf_mc - results[-1,2])/pf_mc*100
print_res = {'Monte Carlo': [N, pf_mc,cov],
             'AK-MCS+U':[results[-1,0], results[-1,2], error]}

for method in print_res:
    print(f'{method}:')
    print(print_res[method])