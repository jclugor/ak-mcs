# %%libraries import
import numpy as np
from scipy.stats import norm

# %% functions
def G(X, k=6):
    g = np.empty((len(X), 4))
    g[:,0] = 3 + 0.1*(X[:,0] - X[:,1])**2 - (X[:,0] + X[:,1])/np.sqrt(2)
    g[:,1] = 3 + 0.1*(X[:,0] - X[:,1])**2 + (X[:,0] + X[:,1])/np.sqrt(2)
    g[:,2] = X[:,0] - X[:,1] + k/np.sqrt(2)
    g[:,3] = X[:,1] - X[:,0] + k/np.sqrt(2)
    return g.min(axis=1)

# %% 1. Generation of a Monte Carlo population in the design space
N = int(1e6)
X = np.random.normal(size=(N,2))

y_p = G(X)

pf = ((y_p <= 0).sum())/N

print(pf)
cov = np.sqrt((1 - pf)/(pf*len(X)))
print(cov)