# Author: Jean-Charles Croix
# Year: 2019
# email: j.croix@sussex.ac.uk

#This third code provides a classification of discrete observations.
#y = (T(t_i),t_i_)
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
    jobid_run = int(np.floor(jobid/60))
    data_idx = int(np.mod(jobid, 60))

    np.random.seed(13*jobid)

    N = 1000 # State space
    N_MC = 3000 # MCMC steps
    # Read the kde estimators
    with open('Priors.pickle_N_SI_all', 'rb') as f:
        Priors = pickle.load(f)

    u_test = np.loadtxt("u_test_N_SI_all.csv")
    test_idx = np.loadtxt("Test_idx_N_SI_all.csv", dtype="int32")[data_idx]
    y = np.loadtxt("../Data/Data_"+str(test_idx)+".csv",
                   delimiter=",",
                   skiprows=1)[:, 2*jobid_run:(2+2*jobid_run)]

    MCT = np.zeros((3, 3000))
    res = np.zeros((3, 3000))
    
    lb = [-5, -5, -5]
    ub = [5, 5, 5]
    
    #lb2 = [1e-7, -2, 0.5]
    #ub2 = [1e-2, 2, 1.5]
    MAP_temp = np.array([])

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

        #mean_u = v_to_u(np.zeros((1,3)), mean, chol)[0, :]

        # Negative log-integrand in the u-space
        def pen_log_like_u(u):
            u = u[np.newaxis, :]
            res = -log_like(u[0, :], y, u_test[data_idx, 3], N)
            res -= kde.score(u_to_v(u, mean, chol_inv))
            return  res
        
#        def pen_log_like_u_levin(u):
#            u = u[np.newaxis, :]
#            res = -log_like_levin(u[0, :], y, u_test[data_idx, 3], N)
#            res -= kde.score(u_to_v(u, mean, chol_inv))
#            return  res
#        
#        def pen_log_like_u_exp(u):
#            u = u[np.newaxis, :]
#            res = -log_like_exp(u[0, :], y, u_test[data_idx, 3], N)
#            res -= kde.score(u_to_v(u, mean, chol_inv))
#            return  res
#        
#        def pen_log_like_u_old(u):
#            u = u[np.newaxis, :]
#            res = -log_like_old(u[0, :], y, u_test[data_idx, 3], N)
#            res -= kde.score(u_to_v(u, mean, chol_inv))
#            return  res

        # Negative log-integrand in the v-space
        def pen_log_like_v(v):
            v = v[np.newaxis, :]
            res = -log_like(v_to_u(v, mean, chol)[0, :], y, u_test[data_idx, 3], N)
            res -= kde.score(v)
            return  res
        
        # MAP point in the u-coordinates
        #min_u, min_fu = pso(pen_log_like_u, lb=lb2, ub=ub2, maxiter=5, swarmsize=100)
        #u_max = minimize(pen_log_like_u, min_u, method="Nelder-Mead")
        #v_MAP = u_to_v(u_max.x[np.newaxis, :], mean, chol_inv)[0, :]
        
        # MAP point in the v-coordinates and its Hessian
        min_v, min_fv = pso(pen_log_like_v, lb=lb, ub=ub, maxiter=5, swarmsize=150)
        v_max = minimize(pen_log_like_v, min_v, method="Nelder-Mead")
        v_MAP = v_max.x
        u_MAP = v_to_u(v_max.x[np.newaxis, :], mean, chol)[0, :]
        MAP_temp = np.hstack((MAP_temp, u_MAP))

        # Computing the Hessian at MAP location
        Hess = nd.Hessian(pen_log_like_v, step=1e-3, method="central")
        neg_Hess = Hess(v_MAP)

        if (np.linalg.det(neg_Hess)<=0):
            Hess = nd.Hessian(pen_log_like_v, step=1e-2, method="central")
            neg_Hess = Hess(v_MAP)

        Cov_max = np.linalg.inv(neg_Hess)

        # MH sampling of posterior distribution
        MCMC_RW = MC(negll = pen_log_like_v, Cov = Cov_max)
        proba, results = MCMC_RW.MCMC(v_MAP, h=0.6, n=500)

        def dist1(v, threshold):
            # Compute maximum distance
            IndA = np.zeros(v.shape[0])
            vc = v-v_MAP
            for (i, vi) in enumerate(vc):
                IndA[i] = np.matmul(vi, np.linalg.solve(Cov_max, vi))<=threshold
            return IndA

        def dist2(v):
            # Compute Ind_A
            dist = np.zeros(v.shape[0])
            vc = v-v_MAP
            for (i, vi) in enumerate(vc):
                dist[i] = np.matmul(vi, np.linalg.solve(Cov_max, vi))
            return dist

        # Region A with P(A|y)=1
        idx = np.where(np.abs(results[:, 0]-results[0, 0])<=3)[0]
        temp = dist2(results[idx, 1:4])
        threshold = temp.max()

        # Assessing P(A)
        sample1 = multivariate_normal.rvs(mean=v_MAP, cov=Cov_max, size=10000)
        idx1 = dist1(sample1, threshold)
        temp = kde.score_samples(sample1)-multivariate_normal.logpdf(sample1, mean=v_MAP, cov=Cov_max)
        PA = np.mean(idx1*np.exp(temp))
        print("PA = ", PA)

        # CAME estimator with importance sampling
        sample = multivariate_normal.rvs(mean=v_MAP, cov=5*Cov_max, size=5*N_MC)
        IndA = dist1(sample, threshold)
        idx = np.where(IndA==1)[0]
        
        def newMCT(v):
            res = -pen_log_like_v(v)
            res -= multivariate_normal.logpdf(v, mean=v_MAP, cov=5*Cov_max)
            return res
        
        for j in range(np.minimum(3000, idx.size)):
            MCT[i, j] = newMCT(sample[idx[j], :])

        res[i, :] = np.log(PA*np.cumsum(np.exp(MCT[i, :]-MCT[i, :].max())) / np.arange(1, 3000+1))+MCT[i, :].max()

    np.savetxt("./Results/MCMC_"+str(jobid)+"_Final.csv", res, delimiter=",")
    np.savetxt("./Results/MAPs_"+str(jobid)+".csv", MAP_temp)

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    #Classify(*sys.argv[1:])
    Classify(1)
