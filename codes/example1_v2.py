# %%libraries import
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.spatial import distance

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

def U2(q10, X, X_D):
    dist = lambda x2: np.apply_along_axis(lambda x1: distance.euclidean(x1, x2), 1, X_D).min()
    distances = np.apply_along_axis(dist, 1, X[q10])
    
    return  q10[(q[q10]**(0.5) + 2/distances).argmin()]

# %% 1. Generation of a Monte Carlo population in the design space
N = int(1e6)
N_range = np.arange(N)
X = np.random.normal(size=(N,2))
y = np.empty(N)

# %% 2. Definition of the initial design of experiments
N1 = 20
DoE = np.random.choice(N_range, N1)
no_DoE = np.setdiff1d(N_range, DoE)
kernel =  C() * RBF([1.0 ,1.0], (1e-50, 1e50))
def optim(fun, init, bounds):
    opt_res = minimize(fun, init, method='L-BFGS-B', bounds=bounds, jac=True, options={'maxiter':50000})
    return opt_res.x, opt_res.fun

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, optimizer=optim, alpha=1e-7)

# %% 3. Computation of the Kriging model according to the DoE
compute_kriging = True
st = []
while True:
    while True:
        if compute_kriging:
            X_g = X[DoE]
            y_g = G(X_g)
            # y_mean = y_g.mean()
            # y_std  = y_g.std()
            # y_g = (y_g - y_mean)/y_std
            gp.fit(X_g, y_g)
        # %% 4. Prediction by Kriging and estimation of the probability of failure
        X_p = np.delete(X, DoE, axis=0)
        (y_p, std_p) = gp.predict(X_p, return_std=True)
        y[DoE] = y_g
        y[no_DoE] = y_p
        # y = y*y_std + y_mean
        pf = ((y <= 0).sum())/len(X)
        # %% 5. Identification of the best next point in S to evaluate on the performance function
        q = np.empty(len(X))
        q[no_DoE] = U(y_p, std_p)
        q[DoE] = 10
        st.append(std_p.max())
        idx_q10 = np.argsort(q)[:10]
        min_q = q.argmin()
        next_x = U2(idx_q10, X, X_g)
        #%% 6
        print(f'{len(DoE)}: {q[min_q]:.6f}, {pf}')
        # if q[next_x]  >= 2:
        if q[min_q] >= 2:
            break
        else:
            DoE = np.append(DoE, next_x)
            no_DoE = np.delete(no_DoE, np.where(no_DoE==next_x))
            compute_kriging = True
    cov = np.sqrt((1 - pf)/(pf*len(X)))
    print('cov:', cov)
    if cov >= 0.05:
        X = np.append(X, np.random.normal(size=(N,2)), axis=0)
        y = np.append(y, np.empty(N))
        no_DoE = np.setdiff1d(np.arange(len(y)), DoE)
        compute_kriging = False
    else:
        break
    

