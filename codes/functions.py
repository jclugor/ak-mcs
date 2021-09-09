# -*- coding: utf-8 -*-
"""
The implementation of the AK-MCS algorithm from "AK-MCS: An active learning
reliability method combining Kriging and Monte Carlo Simulation" by B. Echard
et al (DOI: 10.1016/j.strusafe.2011.01.002), and some functions to plot results.

@author: jclugor
"""
# %% modules import
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
plt.rcParams['figure.dpi'] = 200

# %% 
def ak_mcs(X, G, lfun, N1=12, k=0, std=False):
    """
    Estimates the probability of failure of a given performance function and a
    Monte Carlo Population using the AK-MCS algorithm

    ak_mcs(X, G, lfun, k=0)
    
    Parameters
    ----------
    X : ndarray
        (NxM) array containing a Monte Carlo population of M variables and
        N samples.
    G : function
        Performance function.
    lfun : function
        Learning function to be used by the AK-MCS algorithm.
    N1 : integer
        Size of the initial DoE
    k : integer, optional
        When k > 0, the function returns the estimated value of G(X) with their
        respective DoE every k iterations. The default is 0.
    std : bool, optional
        Whter it should return the standard deviation every k iterations.
        The default is False.

    Returns
    -------
    results: ndarray
         (a*Nx3) array containing the size of the DoE, the value of the learning
         function evaluated at its considered limit, and the estimated
         probability of failure, in that order, at each iteration.  
    plot_DoE: list
         It contains the DoE every k iterations, at up to 9 stages (always
         includes the final values). Not returned if k = 0.      
    plot_y: ndarray
         (a*Nxb) array containing the estimated G(X) every k iterations, at up
         to 9 stages (always includes the final values). Not returned k = 0.
    plot_std: ndarray
         (a*Nxb) array containing the standard deviation of G(X) every k
         iterations, at up to 9 stages (always includes the final values).
         Not returned if k = 0 or std = False.

    """
    print("The program execution has started.")
    # get the size and number of variables of the MC population
    N, n_vars = X.shape
    N_range = np.arange(N)
    # initialize y := G(X)
    y = np.empty(N)

    # %% Definition of the initial design of experiments
    DoE = np.random.choice(N_range, N1)  # initial Design of Experiments
    no_DoE = np.setdiff1d(N_range, DoE)  # points that are not in the DoE
    
    # %% Computation of the Kriging model accordint to the DoE
    # initialization of the kernel. It is an anisotrophic squared-exponential
    # kernel (aka Gaussian)
    kernel =  C() * RBF([1.0]*n_vars, (1e-50, 1e50))
    
    # definition of a custom optimizer. It uses the same algorithm of the default 
    # but with a higher number of maximum iterations, to sort out some numerical issues
    def optim(fun, init, bounds):
        opt_res = minimize(fun, init, method='L-BFGS-B', bounds=bounds,
                           jac=True, options={'maxiter':50000})
        return opt_res.x, opt_res.fun
    
    # definition of the GPR
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                  optimizer=optim, alpha=1e-7)
    
    # the model does not need to be fitted right after adding more points to the
    # Monte Carlo population, so we set this default value that could be changed
    compute_kriging = True
    
    ###########################################################################
    # store the number of points in the DoE, the stopping criteria and the
    # probability of failure at each iteration to plot them
    results = np.array([], dtype='float').reshape(0,3)
    # if k > 0, store the corresponding values of plot_y and plot_DoE
    plot_9grid = k > 0
    if plot_9grid:
        plot_y      = np.array([], dtype='float').reshape(N,0)
        plot_DoE    = []
        if std:
            plot_std = np.array([], dtype='float').reshape(N,0)
        i = 0       # counter
        

    ###########################################################################
    while True:
        while True:
            if compute_kriging:
                X_d = X[DoE]        # select the points of the DoE
                y_d = G(X_d)        # evaluate the performance function
                print(f'Using {len(DoE)} points in the DoE')
                gp.fit(X_d, y_d)    # fit the model to the DoE points evaluation
    
            # %% Prediction by Kriging and estimation of the probability of failure
            # predict the performance function evaluation on the points of the
            # population that are not in the DoE
            X_p = np.delete(X, DoE, axis=0)
            (y_p, std_p) = gp.predict(X_p, return_std=True)
            # build the vector y = G(X) using the calculated and the predicted values
            y[DoE]    = y_d
            y[no_DoE] = y_p
            # estimate the probability of failure
            pf = (y <= 0).mean()
            # %% Identification of the best next point in the MC population to
            # evaluate with the performance function
            learnF, next_x, stop = lfun(y_p, std_p, DoE)
    ###########################################################################
            # store the values
            results = np.row_stack((results, [[len(DoE), learnF[next_x], pf]]))
            if plot_9grid:
                if (i % k) == 0:
                    plot_y = np.column_stack((plot_y, y))
                    plot_DoE.append(DoE)
                    if std:
                        tmp = np.empty(N)
                        tmp[DoE] = 0
                        tmp[no_DoE] = std_p
                        plot_std = np.column_stack((plot_std, tmp))
                i += 1
    ###########################################################################
            #%% Check stopping condition
            if stop:
                break
            else:
            # if the stopping condition is not met, add the point selected by the
            # learning function to the DoE
                DoE = np.append(DoE, next_x)
                no_DoE = np.delete(no_DoE, np.where(no_DoE==next_x))
                compute_kriging = True
        # if the stopping condition is met, calculate the coefficient of variation
        # of the probability of failure
        cov = np.sqrt((1 - pf)/(pf*len(X)))
        # if the c.o.v. is too high, the metamodel is not accurate enough, more
        # points have to be added to the MC population and the procedure is repeated
        if cov >= 0.05:
            X = np.append(X, np.random.normal(size=(N,2)), axis=0)
            y = np.append(y, np.empty(N))
            no_DoE = np.setdiff1d(np.arange(len(y)), DoE)
            compute_kriging = False
        else:
            break

    # adjustment to include the final values of y and the DoE
    if plot_9grid:
        if len(plot_DoE) > 9:
            plot_DoE = plot_DoE[:9]
            plot_y = plot_y[:,:9]
        if len(plot_DoE) == 9:
            plot_DoE[-1] = DoE
            plot_y[:,-1]   = y
        else:
            plot_y = np.column_stack((plot_y, y))
            plot_DoE.append(DoE)

    if not plot_9grid:
        to_return = results
    else:
        if std:
            to_return = (results, plot_DoE, plot_y, plot_std)
        else:
            to_return = (results, plot_DoE, plot_y)

    return to_return

# %%
def plot_lim_vals(results, threshold, fun_name, ex_name=None, save=False):
    """
    Plots # of points in DoE vs limit value of learning function

    Parameters
    ----------
    results : ndarray
        Array of results returned by ak_mcs().
    threshold : float
        Threshold to be met by the limit value of the learning function.
    fun_name : string
        Learning function name.
    ex_name : string, optional
        Exercise name, used in the name of the saved file. The default is None.
    save : bool, optional
        Wheter the produced figure have to be saved or not. The default is False.

    Returns
    -------
    None.

    """
    plt.plot(results[:,0], results[:,1], '.')
    plt.axhline(threshold, 0, 1, ls='--', c='red', lw=1, label='threshold')
    plt.yscale('log')
    plt.xlabel('Number of points in DoE')
    plt.legend()
    plt.ylabel(f'Limit value of the {fun_name}')
    plt.title(f'Limit value of learning function {fun_name} through the iterations')
    if save:
        plt.savefig(f'{ex_name}_{fun_name}_lim_values.png')
    plt.show()
    
# %%
def plot_conv(results, pf_mc, fun_name, ex_name=None, save=False):
    """
    Plots # of points in DoE vs estimated probability of failure. It shows the
    standard deviation of the las 5 values at each point, scaled by a factor of
    20.

    Parameters
    ----------
    results : ndarray
        Array of results returned by ak_mcs().
    pf_mc : float
        Proability of failure estimated by the corresponding MCS.
    fun_name : string
        Learning function name.
    ex_name : string, optional
        Exercise name, used in the name of the saved file. The default is None.
    save : bool, optional
        Wheter the produced figure have to be saved or not. The default is False.

    Returns
    -------
    None.

    """
    k = 10
    plt.plot(results[:,0], results[:,2], '+', label='GPR')
    y_limits = plt.gca().get_ylim()
    plt.gca().set_ylim(y_limits)
    pf_std = np.empty(len(results))
    pf_std[k-1:] = [results[i:i+k,2].std() for i in range(len(pf_std)-k+1)]
    pf_std[:k-1] = pf_std[k-1]
    pf_std *= 20
    plt.fill_between(results[:,0],
                     results[:,2] - pf_std, results[:,2] + pf_std,
                     alpha=0.3)
    plt.axhline(pf_mc, 0, 1, ls='--', c='red', lw=1, label='MCS')
    plt.legend()
    plt.xlabel('Number of points in DoE')
    plt.ylabel('Predicted $p_f$')
    plt.title(f'Convergence of $p_f$ with the learning function {fun_name}')
    if save:
        plt.savefig(f'conv_{ex_name}_{fun_name}.png')
    plt.show()

# %% The following functions should only be used with 2D examples.
# %%
def plot_mc(X, fail_points, pf_mc, ex_name=None, save=False):
    """
    Plots the Monte Carlo population with its corresponding values of G(x) > 0

    Parameters
    ----------
    X : ndarray
        The MC population.
    fail_points : ndarray
        1D array containing data with `bool` type. Indexes of the samples from
        X for which G(x) <= 0
    pf_mc : float
        Proability of failure estimated by the corresponding MCS.
    ex_name : string, optional
        Exercise name, used in the name of the saved file. The default is None.
    save : bool, optional
        Wheter the produced figure have to be saved or not. The default is False.

    Returns
    -------
    None.

    """
    plt.plot(X[~fail_points, 0], X[~fail_points, 1], '.b', ms=1, label='$G(x_i) > 0$')
    plt.plot(X[fail_points, 0], X[fail_points, 1], '.r', ms=1, label='$G(x_i) \leq 0$')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axis('equal')
    lgnd = plt.legend()
    lgnd.legendHandles[0]._legmarker.set_markersize(10)
    lgnd.legendHandles[1]._legmarker.set_markersize(10)
    plt.title(f'Performance function applied to the MC population: $p_f = {pf_mc}$')
    if save:
        plt.savefig(f'mc_{ex_name}.png')
    plt.show()
# %%
def plot_iter(X, plot_y, plot_DoE, ex_name=None, save=False):
    """
    Plots the MC population and its predicted values of G(x) > 0 at 9 stages

    Parameters
    ----------
    X : ndarray
        The MC population.
    plot_DoE: list
        Returned by ak_mcs().
    plot_y: ndarray
        Returned by ak_mcs().
    ex_name : string, optional
        Exercise name, used in the name of the saved file. The default is None.
    save : bool, optional
        Wheter the produced figure have to be saved or not. The default is False.

    Returns
    -------
    None.

    """
    fig, axes = plt.subplots(3,3, figsize=(9, 9), sharey=True, sharex=True)
    for i in range(9):
        ax = axes.flatten()[i]
        fail_points = plot_y[:,i] <= 0
        pf_i = fail_points.mean()
        ax.plot(X[~fail_points, 0], X[~fail_points, 1], '.b', ms=1, label='$G(x_i) > 0$')
        ax.plot(X[fail_points, 0], X[fail_points, 1], '.r', ms=1, label='$G(x_i) \leq 0$')
        ax.plot(X[plot_DoE[i], 0], X[plot_DoE[i], 1], 'x', color='lime', ms=10, label='DoE')
        ax.set_title(f'$n = {len(plot_DoE[i])}$, $p_f = {pf_i}$')
    
    fig.supxlabel('$x_1$')
    fig.supylabel('$x_2$')
    fig.suptitle('Predicted failure points through the iterations')
    handles, labels = ax.get_legend_handles_labels()
    lgnd = fig.legend(handles, labels, loc=(0.08, 0.85))
    lgnd.legendHandles[0]._legmarker.set_markersize(10)
    lgnd.legendHandles[1]._legmarker.set_markersize(10)
    plt.tight_layout()
    if save:
        plt.savefig(f'iter_{ex_name}.png')
    plt.show()

# %% 
def plot_1d(X, G, plot_y, plot_DoE, plot_std, ex_name=None, save=False):
    """
    Plot 1D MC population, its predicted values, DoE and standard deviation.

    Parameters
    ----------
    X : ndarray
        The MC population.
    G : function
        The performance function.
    plot_DoE: list
        Returned by ak_mcs().
    plot_y: ndarray
        Returned by ak_mcs().
    plot_std: ndarray
        Returned by ak_mcs().
    ex_name : string, optional
        Exercise name, used in the name of the saved file. The default is None.
    save : bool, optional
        Wheter the produced figure have to be saved or not. The default is False.

    Returns
    -------
    None.

    """
    fig, axes = plt.subplots(3,3, figsize=(9, 9), sharey=True, sharex=True)
    for i in range(9):
        ax = axes.flatten()[i]
        to_plot = np.column_stack((X, plot_y[:,i], plot_std[:,i]))
        to_plot = to_plot[to_plot[:,0].argsort()]
        ax.plot(X, plot_y[:,i], '.b', ms=1)
        ax.plot(X[plot_DoE[i]], plot_y[plot_DoE[i],i], 'xg', ms=5, label='DoE')
        ax.fill_between(to_plot[:,0], to_plot[:,1] + to_plot[:,2],
                        to_plot[:,1] - to_plot[:,2], alpha=0.3)
        ax.axhline(0, X.min(), X.max(), ls='--', c='red', lw=1, label='G(x) = 0')
        ax.set_title(f'$n = {len(plot_DoE[i])}$')
    
    fig.supxlabel('$x$')
    fig.supylabel('$G(x)$')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.1, 0.87))
    fig.suptitle('Adjustment of the new points in DoE')
    plt.tight_layout()
    if save:
        plt.savefig(f'1d_{ex_name}.png')
    plt.show()
