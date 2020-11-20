import pystan
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from functions import *

########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
x, ca = read_pdb("data/AK/AK.pdb")
x=center_pdb(x[ca])
n_atoms, _ = x.shape

# Read Modes
n_modes = 20
A = read_modes("data/AK/modes/vec.", n_modes=n_modes)[ca]
########################################################################################################
#               NMA DEFORMATION
########################################################################################################

n_modes_fitted = 5
q = np.zeros(n_modes)
q[7:(7+n_modes_fitted)]=np.random.uniform(-150,150,n_modes_fitted)
y_nma=np.zeros(x.shape)
for i in range(n_atoms):
    y_nma[i] = np.dot(q ,A[i]) + x[i]


########################################################################################################
#               MOLECULAR DYNAMICS DEFORMATION
########################################################################################################
k_r = 0.001
k_theta= 0.01
k_lj= 1e-8
d_lj=3
sigma_md=0.05
U_lim= 0.5
r_md=np.mean([np.linalg.norm(x[i] - x[i + 1]) for i in range(n_atoms-1)])
_theta_md=[]
for i in range(n_atoms - 2):
    _theta_md.append(np.arccos(np.dot(x[i] - x[i + 1], x[i + 1] - x[i + 2])
                      / (np.linalg.norm(x[i] - x[i + 1]) * np.linalg.norm(x[i + 1] - x[i + 2]))))
theta_md=np.mean(_theta_md)

y, s_md = md_energy_minimization(y_nma, sigma_md, U_lim, k_r, r_md, k_theta, theta_md, k_lj, d_lj)

########################################################################################################
#               BUILDING DENSITY
########################################################################################################

N = 24
sampling_rate=3.5
gaussian_sigma=2
em_density2 = volume_from_pdb(x, N, sigma=gaussian_sigma, sampling_rate=sampling_rate, precision=0.0001)
em_density = volume_from_pdb(y, N, sigma=gaussian_sigma, sampling_rate=sampling_rate, precision=0.0001)

fig, ax =plt.subplots(1,2)
ax[0].imshow(em_density[int(N/2)])
ax[1].imshow(em_density2[int(N/2)])
fig.savefig("results/input.png")


########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

sm = read_stan_model("md_nma_emmap", build=False)

model_dat = {'n_atoms': n_atoms,
             'n_modes':n_modes_fitted,
             'N':N,
             'em_density':em_density,
             'x0':x,
             'A': A[:,7:(7+n_modes_fitted),:],
             'sigma':200,
             'epsilon':np.max(em_density)/10,
             'mu':np.zeros(n_modes_fitted),
             'sampling_rate':sampling_rate,
             'gaussian_sigma' :gaussian_sigma,
             'halfN': int(N/2),

             'U_init':U_lim,
             's_md':s_md,
             'k_r':k_r,
             'r0':r_md,
             'k_theta':k_theta,
             'theta0':theta_md,
             'k_lj':k_lj,
             'd_lj':d_lj
             }
fit = sm.sampling(data=model_dat, iter=300, warmup=200, chains=4)
la = fit.extract(permuted=True )
q_res = la['q']
lp = la['lp__']
for i in range(n_modes_fitted):
    print(" q value "+str(i+7)+" : "+str(np.mean(q_res[:,i])))
x_res = np.mean(la['x'], axis=0)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0],x[:,1], x[:,2], c='r')
ax.scatter(y[:,0],y[:,1], y[:,2], c='b')
ax.scatter(y_nma[:,0],y_nma[:,1], y_nma[:,2], c='orange')
ax.scatter(x_res[:,0], x_res[:,1], x_res[:,2], marker="x", c='g', s=100)
fig.savefig("results/3d_structures.png")
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
fig.savefig("results/modes_amplitudes.png")
# fig.show()

em_density_res = volume_from_pdb(x_res, N, sigma=gaussian_sigma, sampling_rate=sampling_rate, precision=0.0001)
fig, ax =plt.subplots(1,2)
err_map_init = np.square(em_density -em_density2)[int(N/2)]
err_map_final = np.square(em_density -em_density_res)[int(N/2)]
ax[0].imshow(err_map_init, vmax=np.max(err_map_init), cmap='jet')
ax[1].imshow(err_map_final, vmax=np.max(err_map_init), cmap='jet')
fig.savefig("results/error_map.png")

