# %%libraries import
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm
import matplotlib.pyplot as plt

# %% functions
def plot_mean_std(X):
    plt.plot(X[:,0], X[:,1], '-', X[:,0], X[:,1] + X[:,2], X[:,0], X[:,1] - X[:,2])
    mask = XXX_2[:,2]==0
    plt.plot(XXX_2[:,0][mask], XXX_2[:,1][mask], '*')
    plt.ylim((-150, 150))
    plt.show()
def G(X):
    return X**3 - 10

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
    return abs(y-1)/s

# %% 1. Generation of a Monte Carlo population in the design space
N = int(1e5)
N_range = np.arange(N)
X = np.random.uniform(-5, 5, N)
y = np.empty(N)

# %% 2. Definition of the initial design of experiments
N1 = 3
DoE = np.random.choice(N_range, N1)
no_DoE = np.setdiff1d(N_range, DoE)
kernel =  RBF([1.0], (1e-100, 1e100))
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, alpha=1e-5)
# %% 3. Computation of the Kriging model according to the DoE
compute_kriging = True
st = []
while True:
    while True:
        if compute_kriging:
            X_g = X[DoE]
            y_g = G(X_g)
            XX_g = np.array([X_g, y_g, np.zeros(len(X_g))]).transpose()
            gp.fit(X_g.reshape(-1, 1), y_g)
        # %% 4. Prediction by Kriging and estimation of the probability of failure
        X_p = np.delete(X, DoE, axis=0)
        (y_p, std_p) = gp.predict(X_p.reshape(-1, 1), return_std=True)
        XXX = np.concatenate([XX_g, np.array([X_p, y_p, std_p]).transpose()])
        XXX_2 = XXX[np.argsort(XXX[:,0]),:]
        plot_mean_std(XXX_2)
        y[DoE] = y_g
        y[no_DoE] = y_p
        pf = ((y <= 0).sum())/len(X)
        
        # %% 5. Identification of the best next point in S to evaluate on the performance function
        q = np.empty(len(X))
        q[no_DoE] = EFF(y_p, std_p)
        q[DoE] = -1
        st.append(std_p.max())
        next_x = q.argmax()
        #%% 6
        print(f'{len(DoE)}: {q[next_x]}')
        if q[next_x]  <= 0.001:
            break
        else:
            DoE = np.append(DoE, next_x)
            no_DoE = np.delete(no_DoE, np.where(no_DoE==next_x))
            compute_kriging = True
    cov = np.sqrt((1 - pf)/(pf*len(X)))
    print('cov:', cov)
    if cov >= 0.05:
        X = np.append(X, np.random.normal(size=N), axis=0)
        y = np.append(y, np.empty(N))
        no_DoE = np.setdiff1d(np.arange(len(y)), DoE)
        compute_kriging = False
    else:
        break
