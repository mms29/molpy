import pystan
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from functions import *

# import PDB
x, ca = read_pdb("data/AK/AK.pdb")
x=x[ca]
n_atoms, _ = x.shape

# Read Modes
n_modes = 20
A = read_modes("data/AK/modes/vec.", n_modes=n_modes)[ca]

#Simulate Structure
q = np.zeros(n_modes)
q[7] = +300
y=np.zeros(x.shape)
for i in range(n_atoms):
    y[i] = np.dot(q ,A[i]) + x[i]

# READ STAN MODEL
sm = read_stan_model("nma_1mode")

# fit
model_dat = {'n_atoms': n_atoms,
             'y':y,
             'x0':x,
             'A': A[:,7,:],
             'sigma':200,
             'epsilon':1,
             'mu':0}
fit = sm.sampling(data=model_dat, iter=100, chains=4)
la = fit.extract(permuted=True)
q_res = la['q']
print(" q value : "+str(np.mean(q_res)))
x_res = np.mean(la['x'], axis=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0],x[:,1], x[:,2], c='r')
ax.scatter(y[:,0],y[:,1], y[:,2], c='b')
ax.scatter(x_res[:,0],x_res[:,1], x_res[:,2], c='g', marker="x")
fig.show()

