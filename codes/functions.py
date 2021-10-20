# -*- coding: utf-8 -*-
"""
The implementation of the AK-MCS algorithm:
    
Echard et. al. (2011) - AK-MCS: An active learning reliability method combining 
Kriging and Monte Carlo Simulation. Structural Safety. 33. p. 145-154
https://doi.org/10.1016/j.strusafe.2011.01.002

Authors:
JCLR - Juan Camilo Lugo Rojas      jclugor@unal.edu.co
DAAM - Diego Andrés Alvarez Marín  daalvarez@unal.edu.co

DATE          WHO   WHAT
Oct 13, 2021  JCLR  Algorithm
Oct 13, 2021  DAAM  Comments and readability
"""
# %% modules import
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
plt.rcParams['figure.dpi'] = 200

# %% 
def ak_mcs(random_variables, n_MC, G, learning_fun, N1=12):
    """
    Estimates the probability of failure of a given performance function and a
    Monte Carlo Population using the AK-MCS algorithm

    S, idx_DoE, y_DoE, list_pf_hat, list_gp, list_lf_xstar = \
                          ak_mcs(random_variables, n_MC, G, learning_fun, N1=12)
    
    Parameters
    ----------
    random_variables : list
        list of random variables
    n_MC : integer
        Size of the initial Monte Carlo population
    G : function
        Performance function.
    learning_fun : function
        Learning function to be used by the AK-MCS algorithm (CUALES OPCIONES EXISTEN?)
    N1 : integer, optional
        Size of the initial DoE. The default value is 12.

    Returns
    -------
    S : ndarray
        Monte Carlo population of n_MC points generated from random_variables.
        Each row is a random sample from the design space.
    idx_DoE : ndarray
        index of the points of S that comprise the design of experiments
    y_DoE : ndarray
        performance function evaluated on the DoE
    list_pf_hat : list
        probability of failure estimated at each iteration of the method
    list_gp : list
        GaussianProcessRegressor trained at each iteration of the method
    list_lf_xstar : list
        the learning function evaluated at the best next point at each iteration
        according to the learning criterion
    """
    # %% STAGE 0: Initialization of variables
    # number of random variables
    n = len(random_variables)

    # initial number of points in the design of experiments
    Ni = N1

    # list that contain the outputs
    list_pf_hat   = []
    list_gp       = []
    list_lf_xstar = []

    # Definition of the kernel. 
    # It is an anisotrophic squared-exponential kernel (aka Gaussian)
    kernel =  ConstantKernel() * RBF([1.0]*n, (1e-50, 1e50))
    
    # Definition of the optimizer. 
    def optim(fun, x0, bounds):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        opt = scipy.optimize.minimize(
                        fun, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        jac=True,               # calculate gradient vector
                        options={'maxiter':50000})
        return opt.x, opt.fun
    
    # Definition of the Gaussian Process Regressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
    gp = GaussianProcessRegressor(
                        kernel=kernel, 
                        n_restarts_optimizer=10,
                        optimizer=optim, 
                        alpha=1e-7)

    # %% STAGE 1: Generation of a MC population
    S = np.empty((n_MC, n))
    for i, rv in enumerate(random_variables):
        S[:,i] = rv(n_MC)

    # %% STAGE 2: definition of an initial DoE
    # index of samples that do belong/do not belong to the DoE
    idx        = np.arange(n_MC)    
    idx_DoE    = np.random.choice(idx, N1)   # idx samples belong to the DoE
    idx_no_DoE = np.setdiff1d(idx, idx_DoE)  # idx samples DO NOT belong to the DoE

    # evaluate the points in the DoE
    X_DoE = S[idx_DoE]  # select the points of the DoE
    y_DoE = G(X_DoE)    # evaluate the performance function

    compute_kriging = True
    # %%
    while True:
        while True:
            # %% STAGE 3: computation of the Kriging model according to the DoE
            if compute_kriging:
                gp.fit(X_DoE, y_DoE)
                list_gp.append(gp)
                print(f'Using {Ni} points in the DoE')
            compute_kriging = True              
            
            # %% STAGE 4: prediction by Kriging and estimation of the
            #             probability of failure
            
            # predict the performance function evaluation on the points of the
            # population that are not in the DoE
            X_noDoE            = S[idx_no_DoE]
            y_noDoE, std_noDoE = gp.predict(X_noDoE, return_std=True)
            
            # estimate the probability of failure (Echard et. al, equation 12)
            pf_hat = (np.sum(y_DoE <= 0) + np.sum(y_noDoE <= 0))/n_MC
            list_pf_hat.append(pf_hat)

            # %% STAGE 5: Identification of the best next point in S to evaluate
            #             on the performance function
            # x_ast is the best next point in S to evaluate
            lf_x, idx_x_ast, stop = learning_fun(y_noDoE, std_noDoE, idx_DoE)
            list_lf_xstar.append(lf_x[idx_x_ast])

            # %% STAGE 6: Stopping condition on learning
            # Check stopping condition
            if stop:
                break

            # %% STAGE 7: update the previous design of experiments with the best point
            # add the point selected by the learning function to the DoE
            idx_DoE    = np.append(idx_DoE, idx_x_ast)
            idx_no_DoE = np.delete(idx_no_DoE, np.where(idx_no_DoE==idx_x_ast))            
            
            # update the number of samples in the DoE
            Ni += 1             

            # evaluate x_ast on the true performance function
            x_ast = S[idx_x_ast]
            X_DoE = np.row_stack((X_DoE, x_ast))
            y_DoE = np.append(y_DoE, G(x_ast))
            # end while (return to Stage 3)
        
        # %% STAGE 8: computation of the C.o.V. of the probability of failure
        # calculate the coefficient of variation (C.o.V.) using equation 13
        CoV = np.sqrt((1 - pf_hat)/(pf_hat*n_MC))

        # %% STAGE 9: Update the population
        # if the C.o.V. is too high, the metamodel is not accurate enough, more
        # points have to be added to the MC population and the procedure is repeated
        if CoV >= 0.05:
            S_new = np.empty((n_MC, n))
            for i, rv in enumerate(random_variables):
                S_new[:,i] = rv(n_MC)

            S = np.append(S, S_new, axis=0)
            n_MC *= 2

            idx_no_DoE = np.setdiff1d(np.arange(n_MC), idx_DoE)

            compute_kriging = False
            continue # (return to Stage 4)

        # STAGE 10: End of AK-MCS
        break
    # end while

    return S, idx_DoE, y_DoE, list_pf_hat, list_gp, list_lf_xstar

# %%
def plot_learn_criterion(N1, list_lf_xstar, lf_name, lf_threshold):
    """
    Plots number of points in the DoE vs learning criterion of learning function

    plot_learn_criterion(N1, list_lf_xstar, lf_name, lf_threshold)

    Parameters
    ----------
    N1 : integer
        size of the initial DoE
    list_lf_xstar : list
        the learning function evaluated at the best next point at each iteration
        according to the learning criterion
    lf_name : string
        name of the learning function
    lf_threshold : float
        value at which the stopping condition of the learning function is met

    Returns
    -------
    None.

    """
    n_samples = N1 + np.arange(len(list_lf_xstar))    
    plt.plot(n_samples, list_lf_xstar, '.')
    plt.axhline(lf_threshold, 0, 1, ls='--', c='red', lw=1, label='threshold')
    plt.xlabel('Number of calls to $G$')
    plt.yscale('log')
    plt.ylabel('Learning function value for $x^*$')
    plt.title(f'Variation of the learning function {lf_name} for $x^*$')
    #plt.savefig(f'conv_{ex_name}_{lf_name}.svg')
    plt.legend()
    plt.show()
    # if save:
    #     plt.savefig(f'{ex_name}_{fun_name}_lim_values.png')

    
# %%
def plot_conv_pf(N1, list_pf_hat, pf_MCS, lf_name):
    """   
    Plots the convergence of the estimated probability of failure.

    plot_conv_pf(N1, list_pf_hat, pf_MCS, lf_name)

    Parameters
    ----------
    N1 : integer
        size of the initial DoE
    list_pf_hat : list
        probability of failure estimated at each iteration of the method
    pf_MCS : float
        probability of failure estimated by the Monte Carlo Simulation
    lf_name : string
        name of the learning function

    Returns
    -------
    None.

    """
    n_samples = N1 + np.arange(len(list_pf_hat))    
    plt.plot(n_samples, list_pf_hat, '+', label='GP')
    plt.axhline(pf_MCS, 0, 1, ls='--', c='red', lw=1, label='MCS')
    plt.legend()
    plt.xlabel('Number of calls to $G$')
    plt.ylabel('Predicted $p_f$')
    plt.title(f'Convergence of $p_f$ with the learning function {lf_name}')
    #plt.savefig(f'conv_{ex_name}_{lf_name}.svg')
    plt.show()

# %% The following functions should only be used with 2D examples.
# %%
def plot_mc(S, fail_points, pf_MCS, ex_name=None, save=False):
    """
    Plots the Monte Carlo population with its corresponding values of G(S) > 0

    Parameters
    ----------
    S : ndarray
        The MC population.
    fail_points : ndarray
        Indexes of the samples from S for which G(x) <= 0
    pf_MCS : float
        Proability of failure estimated by the MCS.
    ex_name : string, optional
        Exercise name, used in the name of the saved file. The default is None.
    save : bool, optional
        Wheter the produced figure have to be saved or not. The default is False.

    Returns
    -------
    None.

    """
    plt.plot(S[~fail_points, 0], S[~fail_points, 1], '.b', ms=1, label='$G(x_i) > 0$')
    plt.plot(S[fail_points, 0], S[fail_points, 1], '.r', ms=1, label='$G(x_i) \leq 0$')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axis('equal')
    lgnd = plt.legend()
    lgnd.legendHandles[0]._legmarker.set_markersize(10)
    lgnd.legendHandles[1]._legmarker.set_markersize(10)
    plt.title(f'Performance function applied to the MC population: $p_f = {pf_MCS}$')
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
