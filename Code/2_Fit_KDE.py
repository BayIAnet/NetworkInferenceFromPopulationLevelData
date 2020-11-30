# Author: Jean-Charles Croix
# Year: 2019
# email: j.croix@sussex.ac.uk
# email2: F.Di-Lauro@sussex.ac.uk

# This second code fits a KDE estimator to each network family.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
import pickle
from BD_functions import *

np.random.seed(13)

types = np.dtype([('Type', 'U3'),
                  ('k', 'float'),
                  ('tau', 'float'),
                  ('gamma', 'float')])

Data = np.genfromtxt("Parameters.csv",
                     delimiter=',',
                     skip_header=1,
                     dtype=types)

gamma = Data["gamma"]

cap = np.loadtxt("./1_fitting_cap/CAP_LS_fit.csv")

Learn_ER = np.sort(np.random.choice(np.arange(120), size=100, replace=False))
Test_ER = np.setdiff1d(np.arange(120), Learn_ER)
Learn_Reg = np.sort(np.random.choice(np.arange(120, 240), size=100, replace=False))
Test_Reg = np.setdiff1d(np.arange(120, 240), Learn_Reg)
Learn_BA = np.sort(np.random.choice(np.arange(240, 360), size=100, replace=False))
Test_BA = np.setdiff1d(np.arange(240, 360), Learn_BA)

u_ER = cap[Learn_ER, :]
u_ER_test = cap[Test_ER, :]
u_Reg = cap[Learn_Reg, :]
u_Reg_test = cap[Test_Reg, :]
u_BA = cap[Learn_BA, :]
u_BA_test = cap[Test_BA, :]

test_idx = np.hstack((Test_ER, Test_Reg, Test_BA))

# Take the log(C)
u_ER_log = u_ER.copy()
u_ER_log[:, 0] = np.log(u_ER_log[:, 0])
u_Reg_log = u_Reg.copy()
u_Reg_log[:, 0] = np.log(u_Reg_log[:, 0])
u_BA_log = u_BA.copy()
u_BA_log[:, 0] = np.log(u_BA_log[:, 0])

# Rescaled log-datasets
ER_mean = np.mean(u_ER_log, axis=0)
ER_chol = np.linalg.cholesky(np.cov(u_ER_log, rowvar=False))
ER_chol_inv = np.linalg.inv(ER_chol)

Reg_mean = np.mean(u_Reg_log, axis=0)
Reg_chol = np.linalg.cholesky(np.cov(u_Reg_log, rowvar=False))
Reg_chol_inv = np.linalg.inv(Reg_chol)

BA_mean = np.mean(u_BA_log, axis=0)
BA_chol = np.linalg.cholesky(np.cov(u_BA_log, rowqvar=False))
BA_chol_inv = np.linalg.inv(BA_chol)

v_ER = u_to_v(u_ER, ER_mean, ER_chol_inv)
v_Reg = u_to_v(u_Reg, Reg_mean, Reg_chol_inv)
v_BA = u_to_v(u_BA, BA_mean, BA_chol_inv)

# Kernel density estimation of transformed data
# Graph 1
params = {'bandwidth': np.logspace(-3, 1, 1000)}
grid = GridSearchCV(KernelDensity(), params, cv=10)
grid.fit(v_ER)
kde_ER = grid.best_estimator_

# Graph 2
params = {'bandwidth': np.logspace(-3, 1, 1000)}
grid = GridSearchCV(KernelDensity(), params, cv=10)
grid.fit(v_Reg)
kde_Reg = grid.best_estimator_

# Graph 3
params = {'bandwidth': np.logspace(-3, 1, 1000)}
grid = GridSearchCV(KernelDensity(), params, cv=10)
grid.fit(v_BA)
kde_BA = grid.best_estimator_

gamma_test = np.hstack((gamma[Test_ER], gamma[Test_Reg], gamma[Test_BA]))
u_test = np.vstack((u_ER_test, u_Reg_test, u_BA_test))
u_test = np.hstack((u_test, gamma_test.reshape(-1, 1)))

Dict = {
        "kde_ER":kde_ER,
        "mean_ER":ER_mean,
        "chol_ER":ER_chol,
        "chol_inv_ER":ER_chol_inv,
        "kde_Reg":kde_Reg,
        "mean_Reg":Reg_mean,
        "chol_Reg":Reg_chol,
        "chol_inv_Reg":Reg_chol_inv,
        "kde_BA":kde_BA,
        "mean_BA":BA_mean,
        "chol_BA":BA_chol,
        "chol_inv_BA":BA_chol_inv,
        }

# Plot samples to visualize adjustements
v_sample_ER = kde_ER.sample(1000)
u_ER_sample = v_to_u(v_sample_ER, ER_mean, ER_chol)
v_sample_Reg = kde_Reg.sample(1000)
u_Reg_sample = v_to_u(v_sample_Reg, Reg_mean, Reg_chol)
v_sample_BA = kde_BA.sample(1000)
u_BA_sample = v_to_u(v_sample_BA, BA_mean, BA_chol)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u_ER[:, 0], u_ER[:, 1], u_ER[:, 2], color="blue", label="Learn E-R", s=24)
ax.scatter(u_ER_test[:, 0], u_ER_test[:, 1], u_ER_test[:, 2], color="blue", label="Test E-R", s=24, marker="x")
ax.scatter(u_ER_sample[:, 0], u_ER_sample[:, 1], u_ER_sample[:, 2], color="blue", label="Sample E-R", alpha=0.1)

ax.scatter(u_Reg[:, 0], u_Reg[:, 1], u_Reg[:, 2], color="orange", label="Learn E-R", s=24)
ax.scatter(u_Reg_test[:, 0], u_Reg_test[:, 1], u_Reg_test[:, 2], color="orange", label="Test E-R", s=24, marker="x")
ax.scatter(u_Reg_sample[:, 0], u_Reg_sample[:, 1], u_Reg_sample[:, 2], color="orange", label="Sample Reg", alpha=0.1)

ax.scatter(u_BA[:, 0], u_BA[:, 1], u_BA[:, 2], color="green", label="Learn B-A", s=24)
ax.scatter(u_BA_test[:, 0], u_BA_test[:, 1], u_BA_test[:, 2], color="green", label="Test B-A", s=24, marker="x")
ax.scatter(u_BA_sample[:, 0], u_BA_sample[:, 1], u_BA_sample[:, 2], color="green", label="Sample B-A", alpha=0.1)
ax.set_xlabel('C')
ax.set_ylabel('a')
ax.set_zlabel('p')
plt.legend(loc="best")
