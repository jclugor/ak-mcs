# -*- coding: utf-8 -*-
"""
Learning functions employed in the AK-MCS algorithm

Authors:
JCLR - Juan Camilo Lugo Rojas      jclugor@unal.edu.co
DAAM - Diego Andrés Alvarez Marín  daalvarez@unal.edu.co

DATE          WHO   WHAT
Oct 13, 2021  JCLR  Algorithm
Oct 13, 2021  DAAM  Comments and readability
"""
import numpy as np
from scipy.stats import norm

# %%
def build_output(lf_xnoDOE, idx_DoE):
    """
    builds the vector to be returned by a learning function
    
    lf_x = build_output(lf_xnoDOE, idx_DoE)

    Parameters
    ----------
    lf_xnoDOE : ndarray (1D array, float)
        values obtained by evaluating the learning function in the points that 
        do not belong to the design of experiments
    idx_DoE : ndarray (1D array, int)
        index of the points that conform the design of experiments
    
    Returns
    -------
    lf_x : ndarray (1D array, float)
        values of the learning function stored in the corresponding position 
        of the original population
    """    
    n_MC = len(lf_xnoDOE) + len(idx_DoE)
    idx  = np.arange(n_MC)

    # index of samples that DO NOT belong to the DoE
    idx_no_DoE = np.setdiff1d(idx, idx_DoE)  

    lf_x             = np.empty(n_MC) # initialize the output
    lf_x[idx_DoE]    = np.nan         # assign dummy values
    lf_x[idx_no_DoE] = lf_xnoDOE      # assign actual values in their positions

    return lf_x

# %% learning functions
# %%
def learning_fun_U(m, s, idx_DoE):
    """
    Learning function U

    It aims to determine points that are more likely to cross the threshold, 
    that is, they are either closer to the limit state function or have a large 
    uncertainty.
    
    Proposed in:
    Echard et. al. (2011) - AK-MCS: An active learning reliability method
    combining Kriging and Monte Carlo Simulation. Structural Safety. 33. 
    p. 145-154
    https://doi.org/10.1016/j.strusafe.2011.01.002

    extended_U, idx_x_ast, stop = learning_fun_U(m, s, idx_DoE)

    Parameters
    ----------
    m : ndarray (1D array, float)
        mean given by a Kriging model
    s : ndarray (1D array, float)
        standard deviation given by a Kriging model

    Returns
    -------
    extended_U: ndarray (1D array, float)
        value of the learning function U 
    idx_x_ast: integer
        index of the best next point, that is, the index of the point in the
        population that is expected to best improve training.
    stop: bool
        whether the stopping condition is met or not
    """
    # Learning function (Echard et. al., equation 15)
    U = np.abs(m)/s
    extended_U = build_output(U, idx_DoE)

    # index of best next point
    idx_x_ast = extended_U.nanargmin()

    # the stopping condition is met when min(U) >= 2
    stop = extended_U[idx_x_ast] >= 2

    return extended_U, idx_x_ast, stop

# %%
def learning_fun_EFF(m, s, idx_DoE):
    """
    Expected feasibility function (EFF)

    It gives high feasibility values to those points close to the threshold 
    (around a band of width eps) and selects the point with largest uncertainty.

    Proposed in:
    B.J. Bichon, M.S. Eldred, L.P. Swiler, S. Mahadevan, J.M. McFarland (2008).
    Efficient global reliability analysis for nonlinear implicit performance 
    functions. AIAA Journal, 46, pp. 2459-2468
    https://doi.org/10.2514/1.34321

    extended_EFF, idx_x_ast, stop = learning_fun_EFF(m, s, idx_DoE)

    Parameters
    ----------
    m : ndarray (1D array, float)
        mean given by a Kriging model
    s : ndarray (1D array, float)
        standard deviation given by a Kriging model

    Returns
    -------
    extended_EFF: ndarray (1D array, float)
        value of the learning function EFF
    idx_x_ast: integer
        index of the best next point, that is, the index of the point in the
        population that is expected to best improve training.
    stop: bool
        whether the stopping condition is met or not
    """
    # PDF and CDF of the normal distribution 
    npdf = norm.pdf
    ncdf = norm.cdf

    # parameters of the function when used for reliability problems
    a = 0  # threshold

    # definition of eps
    # NOTE: according to Echard et. al., eps = 2*s**2, however in Bichon 
    # et. al., eps = 2*s. In this implementation we will follow Bichon et. al.
    eps = 2*s

    # Bichon et. al, eq. 17 (it is different to Echard et. al., eq. 14)
    a_m   = (a - m)/s         
    ameps_m = ((a - eps) - m)/s
    apeps_m = ((a + eps) - m)/s
    EFF = (m - a)*(2*ncdf(a_m) - ncdf(ameps_m) - ncdf(apeps_m)) \
            -   s*(2*npdf(a_m) - npdf(ameps_m) - npdf(apeps_m)) \
            + eps*(ncdf(apeps_m) - ncdf(ameps_m))

    extended_EFF = build_output(EFF, idx_DoE)

    # index of best next point
    idx_x_ast = extended_EFF.nanargmax()

    # the stopping condition is met when max(EFF) <= 0.001
    stop = extended_EFF[idx_x_ast] <= 0.001

    return extended_EFF, idx_x_ast, stop

# %%
def learning_fun_H(m, s, idx_DoE):
    """
    Learning function H

    It measures uncertainty based on information entropy.
        
    Proposed in:
    Zhaoyan Lv, Zhenzhou Lu, Pan Wang (2015)
    A new learning function for Kriging and its applications to solve 
    reliability problems in engineering. Computers & Mathematics with 
    Applications. Volume 70, Issue 5, September 2015, p. 1182-1197
    https://doi.org/10.1016/j.camwa.2015.07.004

    extended_H, idx_x_ast, stop = learning_fun_H(m, s, idx_DoE)

    Parameters
    ----------
    m : ndarray (1D array, float)
        mean given by a Kriging model
    s : ndarray (1D array, float)
        standard deviation given by a Kriging model

    Returns
    -------
    extended_H: ndarray (1D array, float)
        value of the learning function H
    idx_x_ast: integer
        index of the best next point, that is, the index of the point in the
        population that is expected to best improve training.
    stop: bool
        whether the stopping condition is met or not

    """
    # PDF and CDF of the normal distribution 
    npdf = norm.pdf
    ncdf = norm.cdf

    # Lv et. al., equation 18
    s_p_m = (2*s + m)/s
    s_m_m = (2*s - m)/s
    H = np.abs(np.log(np.sqrt(2*np.pi)*s + 1/2)*(ncdf(s_m_m) - ncdf(-s_p_m))
                             - ((s - m/2)*npdf(s_m_m) + (s + m/2)*npdf(-s_p_m)))

    extended_H = build_output(H, idx_DoE)

    # index of best next point
    idx_x_ast = extended_H.nanargmax()

    # the stopping condition is met when max(H) <= 0.5
    stop = extended_H[idx_x_ast] <= 0.5

    return extended_H, idx_x_ast, stop
