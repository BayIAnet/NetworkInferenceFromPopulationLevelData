# Author: Jean-Charles Croix
# Year: 2019
# email: j.croix@sussex.ac.uk
# email2: F.Di-Lauro@sussex.ac.uk
# This first code provides the Least-Square estimation of C,a,p model from raw data uk, tk.
# uk is the average SI in state k
# tk is the time spent in state k

import numpy as np
from BD_functions import LS_cap, model_cap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(13)

LS_point = np.zeros((480, 3))

k = np.arange(1001)

for l in range(257,258):
    
    print(l)
    
    Tk = np.loadtxt("./tk/"+str(l)+".csv")
    Uk = np.loadtxt("./uk/"+str(l)+".csv")

    LS_point[l, :] = LS_cap2(Uk, Tk, swarmsize=2000)
    
    ak = Uk / Tk
    plt.figure()
    plt.plot(k, ak, label="ak")
    plt.plot(k, model_cap(LS_point[l, :], k, 1000))
    plt.savefig("ak"+str(l)+".png")
    plt.close()

LS_point = np.savetxt(LS_point, "./1_fitting_cap/CAP_LS_fit.csv")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(LS_point[0:120, 0], LS_point[0:120, 1], LS_point[0:120, 2], color="blue", label="E-R", s=24, marker="o")
ax.scatter(LS_point[120:240, 0], LS_point[120:240, 1], LS_point[120:240, 2], color="orange", label="Reg", s=24, marker="s")
ax.scatter(LS_point[240:360, 0], LS_point[240:360, 1], LS_point[240:360, 2], color="green", label="B-A", s=24, marker="d")
ax.set_xlabel('C')
ax.set_ylabel('a')
ax.set_zlabel('p')
plt.legend(loc="best")
