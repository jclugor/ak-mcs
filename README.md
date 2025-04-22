# AK-MCS Algorithm for Structural Reliability

This project implements the AK-MCS algorithm (Active learning Kriging Monte Carlo Simulation) to estimate structural reliability efficiently.

## Overview

AK-MCS combines Monte Carlo Simulation (MCS) with Kriging (Gaussian Process Regression) and an active learning strategy to reduce the number of performance function evaluations required for accurate failure probability estimation.

## Topics Covered

- **Structural Reliability Concepts**
  - Deterministic, partial probabilistic, and probabilistic approaches
  - Reliability problem formulation: performance function \( G(x) \)
  - Assessment methods: FORM, SORM, and Monte Carlo Simulation

- **Gaussian Process Regression**
  - Conditional Gaussian distributions
  - Kriging with squared-exponential kernel
  - Hyperparameter tuning via log marginal likelihood

- **AK-MCS Algorithm**
  - Initialization with a Monte Carlo sample and Design of Experiments (DoE)
  - Iterative model training and sample point selection using learning functions
  - Stopping criteria based on prediction uncertainty and convergence of the failure probability estimate

- **Learning Functions**
  - **Expected Feasibility Function (EFF)**: Focuses on closeness to the limit state
  - **Function U**: Selects points with high uncertainty near the limit
  - **Function H**: Based on information entropy

- **Application Examples**
  - Simple sinusoidal function
  - Series system with four branches
  - Nonlinear oscillator system

## Code Features

- Python implementation using `scikit-learn` for Kriging
- Modular structure with customizable learning functions
- Performance comparison with classic MCS:
  - Number of function calls
  - Estimated failure probability
  - Coefficient of variation

## References

- Echard, B., Gayton, N., & Lemaire, M. (2011). *AK-MCS: An active learning reliability method combining Kriging and Monte Carlo Simulation*. Structural Safety, 33(2), 145–154.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Melchers, R. E., & Beck, A. T. (2018). *Structural Reliability Analysis and Prediction*. Wiley.
- Additional sources as cited in the project documentation.
