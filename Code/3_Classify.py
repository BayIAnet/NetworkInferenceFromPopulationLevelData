# Author: Jean-Charles Croix
# Year: 2019
# email: j.croix@sussex.ac.uk

# This code provides a classification of discrete observations.
# y = (I(t_i), t_i).

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import numdifftools as nd
from BD_functions import *
from pyswarm import pso
from MCMC import *
import pickle
import sys

def Classify(jobid):
    # Classify all 10 realizations of a graph

    jobid = int(jobid)-1
    jobid_run = int(np.floor(jobid/2))
    data_idx = int(np.mod(jobid, 2))
    
    np.random.seed(13+jobid)

    N = 1000 # State space
    N_MC = 3000 # MCMC steps
    # Read the kde estimators
    with open('Priors.pickle', 'rb') as f:
        Priors = pickle.load(f)
 
    y = np.loadtxt("./Data/Data_"+str(data_idx)+".csv",
                   delimiter=",",
                   skiprows=1)[:, 2*jobid_run:(2+2*jobid_run)]
    MAP_temp = np.array([])
    MCT = np.zeros((3, 3000))
    res = np.zeros((3, 3000))

    lb = [-5, -5, -5]
    ub = [5, 5, 5]

    for i in range(3):
        # For each prior
        print("Prior ", i)
        if (i==0):
            kde = Priors["kde_ER"]
            mean = Priors["mean_ER"]
            chol = Priors["chol_ER"]
            chol_inv = Priors["chol_inv_ER"]
        elif (i==1):
            kde = Priors["kde_Reg"]
            mean = Priors["mean_Reg"]
            chol = Priors["chol_Reg"]
            chol_inv = Priors["chol_inv_Reg"]
        else:
            kde = Priors["kde_BA"]
            mean = Priors["mean_BA"]
            chol = Priors["chol_BA"]
            chol_inv = Priors["chol_inv_BA"]

        # Negative log-integrand
        def pen_log_like(v):
            # u is expected in the transformed space
            v = v[np.newaxis, :]
            res = -log_like_discrete_IL2(v_to_u(v, mean, chol)[0, :], y, 1.00, N)
            res -= kde.score(v)
            return  res

        def newMCT(v):
            res = -pen_log_like(v)
            res -= multivariate_normal.logpdf(v, mean=v_max.x, cov=5*Cov_max)
            return res
        
        # Optimum of integrand and its Hessian
        Test_x, Test_f = pso(pen_log_like, lb, ub, maxiter=5, swarmsize=100)
        v_max = minimize(pen_log_like, Test_x, options={'maxiter':400}, method="Nelder-Mead")
        Hess = nd.Hessian(pen_log_like, step=1e-3, method="central")
        neg_Hess = Hess(v_max.x)
        Cov_max = np.linalg.inv(neg_Hess)
        u_MAP = v_to_u(v_max.x[np.newaxis, :], mean, chol)[0, :]
        MAP_temp = np.hstack((MAP_temp, u_MAP))

        # MH sampling of posterior distribution
        MCMC_RW = MC(negll = pen_log_like, Cov = Cov_max)
        proba, results = MCMC_RW.MCMC(v_max.x, h=0.6, n=500)
        
        def dist1(v, threshold):
            # Compute maximum distance
            IndA = np.zeros(v.shape[0])
            vc = v-v_max.x
            for (i, vi) in enumerate(vc):
                IndA[i] = np.matmul(vi, np.linalg.solve(Cov_max, vi))<=threshold
            return IndA

        def dist2(v):
            # Compute Ind_A
            dist = np.zeros(v.shape[0])
            vc = v-v_max.x
            for (i, vi) in enumerate(vc):
                dist[i] = np.matmul(vi, np.linalg.solve(Cov_max, vi))
            return dist

        # Region A with P(A|y)=1
        idx = np.where(np.abs(results[:, 0]-results[0, 0])<=3)[0]
        temp = dist2(results[idx, 1:4])
        threshold = temp.max()

        # Assessing P(A)
        sample1 = multivariate_normal.rvs(mean=v_max.x, cov=Cov_max, size=10000)
        idx1 = dist1(sample1, threshold)
        temp = kde.score_samples(sample1)-multivariate_normal.logpdf(sample1, mean=v_max.x, cov=Cov_max)
        PA = np.mean(idx1*np.exp(temp))
        print("PA = ", PA)

        # CAME estimator with importance sampling
        sample = multivariate_normal.rvs(mean=v_max.x, cov=5*Cov_max, size=5*N_MC)
        IndA = dist1(sample, threshold)
        idx = np.where(IndA==1)[0]
        
        for j in range(np.minimum(3000, idx.size)):
            MCT[i, j] = newMCT(sample[idx[j], :])

        res[i, :] = np.log(PA*np.cumsum(np.exp(MCT[i, :]-MCT[i, :].max())) / np.arange(1, 3000+1))+MCT[i, :].max()

    np.savetxt("./Results/MCMC_"+str(jobid)+".csv", res, delimiter=",")
    np.savetxt("./Results/MAPs_"+str(jobid)+".csv", MAP_temp)


if __name__ == '__main__':
    # Map command line arguments to function arguments.
    Classify(*sys.argv[1:])
