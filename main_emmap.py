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

#Simulate EM map
n_modes_fitted = 5
q = np.zeros(n_modes)
q[7:(7+n_modes_fitted)]=np.random.uniform(-200,200,n_modes_fitted)
y=np.zeros(x.shape)
for i in range(n_atoms):
    y[i] = np.dot(q ,A[i]) + x[i]
dimX=50
dimY=50
dimZ=50
em_density = np.zeros((dimX,dimY,dimZ))
for i in range(dimX):
    for j in range(dimY):
        for k in range(dimZ):
            s=0
            for a in range(n_atoms):
                s+= np.exp( - ((i -(dimX/2.0)- y[a,0])**2 + (j -(dimY/2.0) - y[a,1])**2 + (k -(dimZ/2.0) - y[a,2])**2))
            em_density[i,j,k] = np.sum(s)
    print(i)


# READ STAN MODEL
sm = read_stan_model("nma_emmap", build=True)

model_dat = {'n_atoms': n_atoms,
             'n_modes':n_modes_fitted,
             'dimX':dimX,
             'dimY':dimY,
             'dimZ':dimZ,
             'em_density':em_density,
             'x0':x,
             'A': A[:,7:(7+n_modes_fitted),:],
             'sigma':200,
             'epsilon':0.001,
             'mu':0}
fit = sm.sampling(data=model_dat, iter=100, warmup=80, chains=4)
la = fit.extract(permuted=True)
q_res = la['q']
for i in range(n_modes_fitted):
    print(" q value "+str(i+7)+" : "+str(np.mean(q_res[:,i])))
# print(" q value 7 : "+str(np.mean(q_res)))
x_res = np.mean(la['x'], axis=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0],x[:,1], x[:,2], c='r')
ax.scatter(y[:,0],y[:,1], y[:,2], c='b')
ax.scatter(x_res[:,0],x_res[:,1], x_res[:,2], c='g', marker="x", s=100)
fig.savefig("results/3d_structures.png")
fig.show()

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(n_modes_fitted):
    parts = ax.violinplot(q_res[:,i],[i], )
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_edgecolor('grey')
    for partname in ('cbars', 'cmins', 'cmaxes', ):
        vp = parts[partname]
        vp.set_edgecolor('#1f77b4')
    ax.plot(i,q[i+7], 'x', color='r')
fig.savefig("results/modes_amplitudes.png")
fig.show()

