# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 22:44:54 2021

@author: jclugor
"""
import numpy as np

n = int(1e3)
# definición de variables
q = np.random.normal(1.15, 0.03335, n)
l = np.random.normal(  60,     0.6, n)
b = np.random.normal(   4,    0.12, n)
h = np.random.normal(   1,    0.03, n)
R = np.random.normal(3600,   298.8, n)

I = (R - (3*q*l**2)/(b*h**2)) < 0
E = np.mean(I)
Var = np.var(I)/n
print(f'Probabilidad de falla: {E:.6f}')
print(f'Varianza: {Var:e}')

