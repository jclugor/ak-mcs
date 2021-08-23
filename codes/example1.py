# -*- coding: utf-8 -*-
"""
Implementation of the Example 1 from section 4 of "AK-MCS: An active learning
reliability method combining Kriging and Monte Carlo Simulation" by B. Echard
et al (DOI: 10.1016/j.strusafe.2011.01.002)

@author: jclugor
"""
# %% modules import
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
import learning_functions as lf

# %% performance function
# a series system of four branches
# it is tested with k=6 and k=7
def G(X, k=6):
    g = np.empty((len(X), 4))
    g[:,0] = 3 + 0.1*(X[:,0] - X[:,1])**2 - (X[:,0] + X[:,1])/np.sqrt(2)
    g[:,1] = 3 + 0.1*(X[:,0] - X[:,1])**2 + (X[:,0] + X[:,1])/np.sqrt(2)
    g[:,2] = X[:,0] - X[:,1] + k/np.sqrt(2)
    g[:,3] = X[:,1] - X[:,0] + k/np.sqrt(2)
    return g.min(axis=1)

# %% Generation of a Monte Carlo population in the design space
N = int(1e6)
N_range = np.arange(N)
# the two considered variables follow a standard normal distribution
X = np.random.normal(size=(N,2))
y = np.empty(N)

# %% Definition of the initial design of experiments
N1 = 12                              # size of the initial design of experiments
DoE = np.random.choice(N_range, N1)  # initial Design of Experiments
no_DoE = np.setdiff1d(N_range, DoE)  # points that are not in the DoE

# %% Computation of the Kriging model accordint to the DoE
# initialization of the kernel. It is an anisotrophic squared-exponential
# kernel (aka Gaussian)
kernel =  C() * RBF([1.0 ,1.0], (1e-50, 1e50))

# definition of a custom optimizer. It uses the same algorithm of the default 
# but with a higher number of maximum iterations, to sort out some numerical issues
def optim(fun, init, bounds):
    opt_res = minimize(fun, init, method='L-BFGS-B', bounds=bounds, jac=True, options={'maxiter':50000})
    return opt_res.x, opt_res.fun

# definition of the GPR
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, optimizer=optim, alpha=1e-7)

# the model does not need to be fitted right after adding more points to the
# Monte Carlo population
compute_kriging = True

#####################################################################################
# store the number of points in the DoE, the stopping criteria and the
# probability of failure at each iteration to plot them
plot_values = np.empty((1,3))
plot_y      = np.empty((N, 9))
i = 0       # counter      
k = 13      # save a copy each k iterations
#####################################################################################
while True:
    while True:
        if compute_kriging:
            X_d = X[DoE]        # select the points of the DoE
            y_d = G(X_d)        # evaluate the performance function
            gp.fit(X_d, y_d)    # fit the model to the DoE points evaluation

        # %% Prediction by Kriging and estimation of the probability of failure
        # predict the performance function evaluation on the points of the
        # population that are not in the DoE
        X_p = np.delete(X, DoE, axis=0)
        (y_p, std_p) = gp.predict(X_p, return_std=True)
        # construct the vector y = G(X) using the calculated and the predicted values
        y[DoE]    = y_d
        y[no_DoE] = y_p
        # estimate the probability of failure
        pf = ((y <= 0).sum())/len(X)
        # %% Identification of the best next point in the MC population to
        # evaluate on the performance function
        learnF, next_x, stop = lf.EFF(y_p, std_p, DoE)
#####################################################################################
        # store the values
        plot_values = np.append(plot_values, [[len(DoE), learnF[next_x], pf]], axis=0)
        if (i % k) == 0: plot_y[:, i//k] = y
        i += 1
#####################################################################################
        #%% Check stopping condition
        if stop:
            break
        else:
        # if the stopping condition is not met, add the point determined by the
        # learning function to de DoE
            DoE = np.append(DoE, next_x)
            no_DoE = np.delete(no_DoE, np.where(no_DoE==next_x))
            compute_kriging = True
    # if the stopping condition is met, calculate the coefficient of variation
    # of the probability of failure
    cov = np.sqrt((1 - pf)/(pf*len(X)))
    # if the c.o.v. is too high, whe metamodel is not accurate enough, more
    # points have to be added to the MC population and the procedure is repeated
    if cov >= 0.05:
        X = np.append(X, np.random.normal(size=(N,2)), axis=0)
        y = np.append(y, np.empty(N))
        no_DoE = np.setdiff1d(np.arange(len(y)), DoE)
        compute_kriging = False
    else:
        break
plot_y[:,-1] = y   # to plot the actual results at the end
# %% results
plt.rcParams['figure.dpi'] = 200
# %% performance function applied to the whole population
fail_points = G(X) <= 0
plt.plot(X[~fail_points, 0], X[~fail_points, 1], '.b', ms=1, label='$p_f > 0$')
plt.plot(X[fail_points, 0], X[fail_points, 1], '.r', ms=1, label='$p_f \leq 0$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
lgnd = plt.legend()
lgnd.legendHandles[0]._legmarker.set_markersize(10)
lgnd.legendHandles[1]._legmarker.set_markersize(10)
plt.title(f'Performance function applied to the MC population: $p_f = {pf}$')

# %% predicted failure through the iterations
fig, axes = plt.subplots(3,3, figsize=(9, 9), sharey=True, sharex=True)
for i in range(9):
    ax = axes.flatten()[i]
    fail_points = plot_y[:,i] <= 0
    pf_i = (fail_points).sum()/len(y)
    ax.plot(X[~fail_points, 0], X[~fail_points, 1], '.b', ms=1, label='$p_f \geq 0$')
    ax.plot(X[fail_points, 0], X[fail_points, 1], '.r', ms=1, label='$p_f < 0$')
    ax.set_title(f'$p_f = {pf_i}$')

fig.supxlabel('$x_1$')
fig.supylabel('$x_2$')
fig.suptitle('Predicted failure points through the iterations')
handles, labels = ax.get_legend_handles_labels()
lgnd = fig.legend(handles, labels, loc='lower right')
lgnd.legendHandles[0]._legmarker.set_markersize(10)
lgnd.legendHandles[1]._legmarker.set_markersize(10)
plt.tight_layout()

# %% plot minima values of learning function 
plt.plot(plot_values[:,0], plot_values[:,1], '.')
plt.xlabel('Number of points in DoE')
plt.ylabel('Minimum value of the learning function')
plt.title('Minimum value of learning function EFF through the iterations')
plt.show()

# %% plot convergence of pf

plt.plot(plot_values[:,0], plot_values[:,2], '.')
plt.xlabel('Number of points in DoE')
plt.ylabel('Predicted $p_f$')
plt.title('Convergende of $p_f$')
plt.show()
