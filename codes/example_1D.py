# -*- coding: utf-8 -*-
"""
Example to illustrate how does the method select the points to be added to the DoE

Author:
JCLR - Juan Camilo Lugo Rojas      jclugor@unal.edu.co
"""
# %% modules import
import numpy as np
import learning_functions as lf
from functions import ak_mcs, mc_population, plot_1d

# %% performance function
def G(X):
    return np.sin(X).flatten()

# %% Generation of a Monte Carlo population in the design space
n_MC = 10_000       # number of points in the MC population

random_variables = [ lambda n: np.random.normal(0, 2, n)]
S = np.random.normal(0, 2, size=(n_MC,1))

# %% solution by MCS
seed = 2222  # random seed to generate the MC population
np.random.seed(seed=seed)
S = mc_population(random_variables, n_MC)
fail_points = G(S) <= 0
pf_MCS      = fail_points.mean()
CoV         = np.sqrt((1-pf_MCS)/(pf_MCS*n_MC))

# %% computation of the AK-MCS algorithm
N1 = 5     # size of the initial DoE

np.random.seed(seed=seed)
idx_DoE, y_DoE, list_pf_hast, list_gp, list_lf_xstar = \
                   ak_mcs(random_variables, n_MC, G, lf.learning_fun_U, N1)


# %% results
plot_1d(S, G, idx_DoE, list_gp, N1)
# error = (pf_mc - results[-1,2])/pf_mc*100
# print_res = {'Monte Carlo': [n_MC, pf_mc,cov],
#              'AK-MCS+U':[results[-1,0], results[-1,2], error]}

# for method in print_res:
#     print(f'{method}:')
#     print(print_res[method])