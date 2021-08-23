# -*- coding: utf-8 -*-
"""
Here we have some learning functions to use with the AK-MCS algorithm.

@author: jclugor
"""
import numpy as np
from scipy.stats import norm

# %% auxiliar functions
def build_output(lf, DoE, DoE_value):
    """
    builds the vector to be returned by a learning function
    
    build_output(lf, DoE, DoE_value)

    Parameters
    ----------
    lf  : ndarray
        1D array containing data with `float` type. It corresponds to the values
        obtained by evaluating the learning function in the points not in the
        design of experiments.
    DoE : ndarray
        1D array containing data with `int` type. The points in the design of
        experiments
    DoE_value : float
        dummy value to be assigned in the positions where the learning function
        is not evaluated

    Returns
    -------
    out_vector :   ndarray
        1D array containing data with `float` type. Values from `lf` in their
        position in the original population
    """    
    length = len(lf) + len(DoE)                 # population length
    no_DoE = np.setdiff1d(range(length), DoE)   # idx of points not in the DoE
    out_vector = np.zeros(length)               # initialize the output
    out_vector[DoE]   = DoE_value               # assign dummy values
    out_vector[no_DoE] = lf                     # assign actual values in their positions

    return out_vector

# %% learning functions
#%%
def U(m, s, DoE):
    """
    Learning function U
    
    U(m, s)
    
    Proposed by B. Echard, N. Gayton and M. Lemaire (DOI:10.1016/j.strusafe.2011.01.0020).
    It aims to determine points that are more likely to cross the threshold.

    Parameters
    ----------
    m : ndarray
        1D array containing data with `float` type. It corresponds to the mean
        given by a Kriging model.
    s : ndarray
        1D array containing data with `float` type. It corresponds to the
        standard deviation given by a Kriging model.

    Returns
    -------
    U:   ndarray
        1D array containing data with `float` type. The values of the function
        evaluated
    idx: integer
        index of the point in the population that is expected to improve the
        estimation the most.
    stop_condition: bool
        wheter the stopping condition is met or not

    """
    U = abs(m)/s
    # the stopping condition is met when min(U) >= 2
    out_vector = build_output(U, DoE, 10)  # assign a dummy value of 10
    idx = out_vector.argmin()
    return out_vector, idx, out_vector[idx] >= 2

# %%
def EFF(m, s, DoE):
    """
    Expected feasibility function
    
    EFF(m, s)
    
    Proposed by B. Bichon et al (DOI: 10.2514/1.34321). It gives high
    feasibility values to points close to the threshold and points with large
    uncertainty.

    Parameters
    ----------
    m : ndarray
        1D array containing data with `float` type. It corresponds to the mean
        given by a Kriging model.
    s : ndarray
        1D array containing data with `float` type. It corresponds to the
        standard deviation given by a Kriging model.

    Returns
    -------
    EFF: ndarray
        1D array containing data with `float` type. It is the function
        evaluated.
    idx: integer
        index of the point in the population that is expected to improve the
        estimation the most.
    stop_condition: bool
        wheter the stopping condition is met or not
    """
    # PDF and CDF of the normal distribution 
    npdf = norm.pdf
    ncdf = norm.cdf

    # parameters of the function when used for reliability problems
    eps = 2 * s
    a = 0          # threshold

    m_a   = (a - m)/s         
    m_a_m = (a - eps - m)/s    # m_a+
    m_a_p = (a + eps - m)/s    # m_a-

    EFF = (m - a) * (2*ncdf(m_a) - ncdf(m_a_m) - ncdf(m_a_p)) \
              - s * (2*npdf(m_a) - npdf(m_a_m) - npdf(m_a_p)) \
             +eps * (ncdf(m_a_p) - ncdf(m_a_m))

    # the stopping condition is met when max(EFF) <= 0.001
    out_vector = build_output(EFF, DoE, -1)  # assign a dummy value of -1
    idx = out_vector.argmax()
    return out_vector, idx, out_vector[idx] <= 0.001