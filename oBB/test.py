#!/usr/bin/env python

from __future__ import division
import numpy as np
from obb import obb
import scipy.optimize
from scipy.special import lambertw as W
from numpy import sin, cos, diag, ones, zeros


def f(p):

    global alpha, lam

    rev = 0
    for a, l in zip(alpha, lam):
        rev += l * np.sum(p*np.exp(a-p) / (1 + np.sum(np.exp(a-p))))
    
    return - rev


def g(p):

    global alpha, lam

    grad = 0
    for a, l in zip(alpha, lam):
        q = np.exp(a - p) / (1 + np.sum(np.exp(a - p)))
        grad += l * q * (1 - p + np.sum(q*p))
    return - grad


def bndH(_, __):
    
    global U, L
    return (-U, -L)

# Input Settings
# Algorithm (T1, T2_individual, T2_synchronised, T2_synchronised_rr)
alg = 'T1'

# Model type (q - norm quadratic, g/Hz/lbH/E0/Ediag - min eig. quadratic,
# c - norm cubic, gc - gershgorin cubic)
mod = 'q'

# Tolerance
tol = 1e-2


n = 2
m = 2
# alpha = np.random.uniform(1, 5, size=(m,n))
alpha = np.asarray([[3,1],[12,6]])
lam = np.asarray([0.8, 0.2])

# lam = np.random.uniform(0,1,size=m)
# lam = lam / np.sum(lam)

# alpha = np.array([1,5])

maxsum = np.max([np.sum(np.exp(a-1)).real for a in alpha])
pmax = 1 + W(maxsum).real
xi = [(pmax - 1) / maxsum * np.min([np.sum(np.exp(a[np.arange(n)!=j])) for a in alpha]) for j in range(n)]
pmin = np.asarray([W(np.exp(np.min(alpha[:,j]) - xi[j] - 1)).real + xi[j] + 1 for j in range(n)])

# Run scipy opt
out = scipy.optimize.minimize(f, [9, 9])

# Bounds
U = np.zeros((n,n)) + np.nan

for j in range(n):
    u = 0
    for a, lamc in zip(alpha, lam):
        u += lamc * np.max([W(np.sum(np.exp(a-1))).real,
                            W(np.exp(a[j]-1) / np.sum(np.exp(a[np.arange(n)!=j] - pmax))).real])
    U[j,j] = u

for j in range(n):
    for k in range(n):
        if j != k:
            u = 0
            for a, lamc in zip(alpha, lam):
                u += lamc * W(np.sum(np.exp(a-1))).real
            U[j,k] = u

L = np.zeros((n,n)) + np.nan

for j in range(n):
    L[j,j] = - U[j,j] - 1

for j in range(n):
    for k in range(n):
        if j != k:
            u = 0
            for a, lamc in zip(alpha, lam):
                u += - lamc * (W(np.exp(a[j]-1) / np.sum(np.exp(a[np.arange(n) != j] - pmax))).real +
                               W(np.exp(a[k]-1) / np.sum(np.exp(a[np.arange(n) != k] - pmax))).real)
            L[j,k] = u 


# Run oBB
xs, fxs, tol, itr = obb(f, g, 0, bndH, 0, pmin, np.asarray([pmax]*n), alg, mod, tol=tol)

print(out)

