# Author: Jean-Charles Croix
# Year: 2019
# email: j.croix@sussex.ac.uk

# This set of functions provide likelihood related functions.

import numpy as np
from scipy.linalg import pascal
from pyswarm import pso
from matplotlib import pyplot as plt

# Transformation of datasets
def u_to_v(u, mean, chol_inv):
    v = u.copy()
    v[:, 0] = np.log(v[:, 0])
    v = np.matmul(v - mean, chol_inv.transpose())
    return v


def v_to_u(v, mean, chol):
    u = v.copy()
    u = np.matmul(u, chol.transpose()) + mean
    u[:, 0] = np.exp(u[:, 0])
    return u


def model_cap(u, k, N):
    return u[0] * k**u[2] * (N-k)**u[2] * (u[1]*(k-N/2)+1)


# Least-Squares functional
def LS2(u, Uk, Tk):
    idx = np.where(Tk>0)[0]
    N = Tk.size-1
    k = np.arange(0, N+1, dtype=int)
    ak = model_cap(u, k, N)
    ak_hat = Uk[idx]/Tk[idx]
    temp = ak[idx]-ak_hat
    dist = np.dot(temp, temp)
    return dist


# Log-likelihood using Inverse Laplace transform and continued fractions
def log_like_discrete_IL2(u, Data, gamma, N):
    k = np.arange(0, N+1, dtype=int)
    ak = model_cap(u, k, N)
    ck = gamma * np.arange(N+1)
    log_like = 0
    for i in range(1, Data.shape[0]):
        proba = p_k0k_vec(Data[i, 0]-Data[i-1,0], Data[i, 1], Data[i-1, 1], ak, ck)
        log_like += np.log(proba)
    return log_like


def cont_frac_vec(a, b, eps=1e-30):
    # Compute the continued fraction (Modified Lentz's method)
    # a0/(b0+)a1/(b1+)a2/(b2+)
    D = np.zeros(b.shape[0], dtype=complex)
    frac = eps * np.ones(b.shape[0], dtype=complex)
    C = frac
    for i in range(a.size):
        D = 1 / (b[:, i]+a[i]*D)
        C = b[:, i] + a[i]/C
        frac *= C * D
    return frac


def f_vec(s, k, k0, ak, ck):
    # Compute the Laplace transform of P(I(t)=k|I(0)=k0,theta).
    # s is an array of complex values

    log_cte = 0
    if (k<k0):
        log_cte = np.sum(np.log(ck[(k+1):(k0+1)]))
    elif (k>k0):
        log_cte = np.sum(np.log(ak[k0:k]))

    idx1 = np.minimum(k, k0)
    idx2 = np.maximum(k, k0)

    # Compute B_idx1(s), B_{idx2}(s), B_{idx2+1}(s)
    a = np.ones(ak.size) #a_1,  a_2, ..., a_N
    b = np.ones((s.size, ak.size), dtype=complex)
    a[1:] = -ak[0:-1] * ck[1:]
    b = b * s[:, np.newaxis]
    b = b + ak + ck
    cf = cont_frac_vec(a[(idx2+1):], b[:, (idx2+1):])

    D = np.zeros((s.size, idx2+2), dtype=complex) #D_0, D_1,...,D_{max(k0, k)+1}
    for i in range(idx2+1):
        D[:, i+1] = 1 / (b[:, i]+a[i]*D[:, i])

    # Compute f_{k0, k}(s)
    log_res = np.sum(np.log(D[:, (idx1+1):(idx2+2)]), axis=1)
    res = np.exp(log_res+log_cte) / (1+D[:, idx2+1]*cf)
    return res


def p_k0k_vec(t, k, k0, ak, ck, gamma=4, N=30):
    # Compute P(I(t)=k|I(0)=k0,theta) by Crawford method
    # Inverse Laplace transform of continued fraction representation
    # Euler's transform
    A = gamma*np.log(10)
    idx = np.arange(0, N)
    s = (A+2*idx*np.pi*1j)/2/t
    val = np.real(f_vec(s, int(k), int(k0), ak, ck))
    col = np.ones((N, N), dtype="int")*np.arange(N)
    lig = np.ones((N, N), dtype="int")*np.arange(N)[:, np.newaxis]
    temp = np.triu(np.choose(col-lig, val, mode="clip"))*(-1)**(col+lig)
    temp *= pascal(N, kind='upper')/2**np.arange(1, N+1)
    res = np.exp(A/2) / t * (np.sum(temp)-val[0]/2)
    return np.maximum(res, 1e-100)

# Find the Least-Square estimator using Pyswarm
def LS_cap(Uk, Tk, swarmsize=1000, maxiter=500):
    # Compute MLE estimator from Uk, Tk for model 2

    def temp(v):
        return LS2(v, Uk, Tk)

    N = Tk.size
    LB = np.array([0, -2/N, 0.1])
    UB = np.array([2, 2/N, 4])
    # Constrained optimization
    opt = pso(temp, lb=LB, ub=UB, swarmsize=swarmsize, maxiter=maxiter)
    
    return opt[0]
