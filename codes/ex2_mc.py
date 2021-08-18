
# %%libraries import
import numpy as np
from scipy.stats import norm

# %% functions
def G(X):
    return (10 - X[:,1]**2 + 5*np.cos(2*np.pi*X[:,1]) 
               - X[:,0]**2 + 5*np.cos(2*np.pi*X[:,0]))
        

# %% 1. Generation of a Monte Carlo population in the design space
N = int(6e4)
X = np.random.normal(size=(N,2))

y_p = G(X)

pf = ((y_p <= 0).sum())/N

print(pf)
cov = np.sqrt((1 - pf)/(pf*len(X)))
print(cov)