# %%libraries import
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

# %% functions
def G(X_g, k=7):
    g = np.empty((len(X_g), 4))
    g[:,0] = 3 + 0.1*(X_g[:,0] - X_g[:,1])**2 - (X_g[:,0] + X_g[:,1])/np.sqrt(2)
    g[:,1] = 3 + 0.1*(X_g[:,0] - X_g[:,1])**2 + (X_g[:,0] + X_g[:,1])/np.sqrt(2)
    g[:,2] = X_g[:,0] - X_g[:,1] + k/np.sqrt(2)
    g[:,3] = X_g[:,1] - X_g[:,0] + k/np.sqrt(2)
    return g.min(axis=1)

def EFF(y, s):
    npdf = norm.pdf
    ncdf = norm.cdf
    eps = 2 * s; a = 0
    m1 = (a - y)/s
    m2 = (a - eps - y)/s
    m3 = (a + eps - y)/s
    return ((y - a)*(2*ncdf(m1) - ncdf(m2) - ncdf(m3))
              - s*(2*npdf(m1) - npdf(m2) - npdf(m3))
              + eps*(ncdf(m3)- ncdf(m2)))

def U(y, s):
    return abs(y)/s
# %% 1. Generation of a Monte Carlo population in the design space
N = int(1e6)
X = np.random.normal(size=(N,2))
y = np.empty(len(X))

# %% 2. Definition of the initial design of experiments
N1 = 12
DoE = np.random.choice(np.arange(N), N1)
kernel =  C() * RBF(length_scale=[1.0 ,1.0])
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, alpha=1e-5)
# %% 3. Computation of the Kriging model according to the DoE
compute_kriging = True
while True:
    while True:
        if compute_kriging:
            X_d = X[DoE]
            y_d = G(X_d)
            gp.fit(X_d, y_d)
        # %% 4. Prediction by Kriging and estimation of the probability of failure
        X_p = np.delete(X, DoE, axis=0)
        (y_p, std_p) = gp.predict(X_p, return_std=True)
        y[DoE] = y_d
        idx_p = np.setdiff1d(np.arange(len(X)), DoE)
        y[idx_p] = y_p
        pf = ((y <= 0).sum())/len(X)
        
        # %% 5. Identification of the best next point in S to evaluate on the performance function
        q = U(y_p, std_p)
        q2 = q.copy()
        q2[DoE] = 10
    
        idx_max = q2.argmin()
        # idx_min = np.delete(q, DoE).argmin()
        #%% 6
        print(f'{len(DoE)}: {q[idx_max]}')
        # if q[idx_max] <= 0.001:
        #     break
        if q[idx_max]  >= 2:
            break
        else:
            DoE = np.append(DoE, idx_max)
            compute_kriging = True
    cov = np.sqrt((1 - pf)/(pf*len(X)))
    print('cov:', cov)
    if cov >= 0.05:
        X = np.append(X, np.random.normal(size=(N,2)), axis=0)
        y = np.append(y, np.empty(len(X)))
        compute_kriging = False
    else:
        break
    