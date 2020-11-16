import pystan
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from functions import *

# import PDB
x, ca = read_pdb("data/AK/AK.pdb")
x=center_pdb(x[ca])
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

dimX = 24
dimY = 24
dimZ = 24
em_density = volume_from_pdb(y, size=(dimX,dimY,dimZ), sigma=1, sampling_rate=2, precision=0.001)
em_vector = to_vector(em_density)

plt.imshow(em_density[16])

# READ STAN MODEL
n_shards=1
os.environ['STAN_NUM_THREADS'] = str(n_shards)
sm = read_stan_model("nma_emmap_map_rect", build=True, threads=1)

model_dat = {'n_atoms': n_atoms,
             'n_modes':n_modes_fitted,
             'dimX':dimX,
             'dimY':dimY,
             'dimZ':dimZ,
             'em_density':em_vector,
             'x0':x,
             'A': A[:,7:(7+n_modes_fitted),:],
             'sigma':200,
             'epsilon':0.1,
             'mu':0,
             'sampling_rate':2,
             'gaussian_sigma' :1,
             'center_transform': np.array([dimX/2,dimY/2,dimZ/2]),
             'n_shards' : n_shards}
fit = sm.sampling(data=model_dat, iter=20, warmup=10, chains=3)
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

