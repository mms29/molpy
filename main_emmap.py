import pystan
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from functions import *

# import PDB
x, ca = read_pdb("/home/guest/Downloads/AK.pdb")
x=x[ca]
n_atoms, _ = x.shape

# Read Modes
n_modes = 20
A = read_modes("/home/guest/ScipionUserData/projects/HEMNMA-3D-Remi/Runs/000074_FlexProtNMA/modes/vec.", n_modes=n_modes)[ca]

#Simulate EM map
n_modes_fitted = 2
q = np.zeros(n_modes)
q[7:(7+n_modes_fitted)]=np.random.uniform(-200,200,n_modes_fitted)
y=np.zeros(x.shape)
for i in range(n_atoms):
    y[i] = np.dot(q ,A[i]) + x[i]
dimX=50
dimY=50
dimZ=50
dimBase=50
sigma=2
em_density = np.zeros((dimX,dimY,dimZ))
for i in range(dimX):
    for j in range(dimY):
        for k in range(dimZ):
            s=0
            for a in range(n_atoms):
                s+= np.exp( - (((i -(dimX/2.0))*(dimBase/dimX) - y[a,0])**2 + ((j -(dimY/2.0))*(dimBase/dimY) - y[a,1])**2 + ((k -(dimZ/2.0))*(dimBase/dimZ) - y[a,2])**2))
                # s += (1 / ((2 * np.pi * (sigma ** 2)) ** (3 / 2)) * np.exp((-1 / (2 * (sigma ** 2))) * (np.linalg.norm(np.array([i - int(dimX / 2), j - int(dimY / 2), k - int(dimZ / 2)] - x[a])) ** 2)))
            em_density[i,j,k] = np.sum(s)


# READ STAN MODEL
sm = read_stan_model("nma_emmap", build=True)


