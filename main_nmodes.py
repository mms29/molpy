import pystan
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from functions import *

# import PDB
x, ca = read_pdb("data/AK/AK.pdb")
x=x
n_atoms, _ = x.shape

# Read Modes
n_modes = 20
A = read_modes("data/AK/modes/vec.", n_modes=n_modes)

#Simulate Structure
n_modes_fitted = 10
q = np.zeros(n_modes)
q[7:(7+n_modes_fitted)]=np.random.uniform(-200,200,n_modes_fitted)
y=np.zeros(x.shape)
for i in range(n_atoms):
    y[i] = np.dot(q ,A[i]) + x[i]

# READ STAN MODEL
n_shards=1
os.environ['STAN_NUM_THREADS'] = str(n_shards)
sm = read_stan_model("nma_nmodes", build=False)

model_dat = {'n_atoms': n_atoms,
             'n_modes':n_modes_fitted,
             'y':y,
             'x0':x,
             'A': A[:,7:(7+n_modes_fitted),:],
             'sigma':200,
             'epsilon':1,
             'mu':0,
             'n_shards':n_shards}

fit = sm.sampling(data=model_dat, iter=1000, chains=4)
la = fit.extract(permuted=True)
q_res = la['q']
for i in range(n_modes_fitted):
    print(" q value "+str(i+7)+" : "+str(np.mean(q_res[:,i])))
# print(" q value 7 : "+str(np.mean(q_res)))
# x_res = np.mean(la['x'], axis=0)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x[:,0],x[:,1], x[:,2], c='b')
# # ax.scatter(y[:,0],y[:,1], y[:,2], c='b')
# ax.scatter(y[:,0],y[:,1], y[:,2], c='r')
# ax.scatter(x_res[:,0],x_res[:,1], x_res[:,2], c='g', marker="x", s=100)
# ax.legend(["Initial Structure", "Ground Truth","NMA fit"])
# fig.show()


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
ax.set_xlabel("modes")
ax.set_ylabel("amplitude")
ax.legend(["Ground Truth"])
fig.suptitle("Normal modes amplitudes distribution")
fig.savefig("results/modes_amplitudes.png")
fig.show()