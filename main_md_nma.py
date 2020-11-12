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
x=x[ca]
n_atoms, _ = x.shape

# Read Modes
n_modes = 20
A = read_modes("data/AK/modes/vec.", n_modes=n_modes)[ca]

########################################################################################################
#               NMA DEFORMATION
########################################################################################################

n_modes_fitted = 10
q = np.zeros(n_modes)
q[7:(7+n_modes_fitted)]=np.random.uniform(-200,200,n_modes_fitted)
y=np.zeros(x.shape)
for i in range(n_atoms):
    y[i] = np.dot(q ,A[i]) + x[i]


########################################################################################################
#               MOLECULAR DYNAMICS DEFORMATION
########################################################################################################
y_md = np.array(y)
k_r = 0.001
k_theta= 0.01
k_lj=1e-8
d_lj=3
s_md = 0
sigma_md=0.05
n_md=0
U_lim= 0.5
r_md=np.mean([np.linalg.norm(x[i] - x[i + 1]) for i in range(n_atoms-1)])
_theta_md=[]
for i in range(n_atoms - 2):
    _theta_md.append(np.arccos(np.dot(x[i] - x[i + 1], x[i + 1] - x[i + 2])
                      / (np.linalg.norm(x[i] - x[i + 1]) * np.linalg.norm(x[i + 1] - x[i + 2]))))
theta_md=np.mean(_theta_md)

#compute current potential
U_bonds_init=0
for i in range(n_atoms - 1):
    r = np.linalg.norm(y[i] - y[i + 1])
    U_bonds_init += k_r * (r - r_md) ** 2

U_angles_init=0
for i in range(n_atoms-2):
    theta = np.arccos(np.dot(y[i] - y[i+1], y[i+1] - y[i+2])
                      / (np.linalg.norm(y[i] - y[i+1]) * np.linalg.norm(y[i+1] - y[i+2])))
    U_angles_init += k_theta * (theta - theta_md) ** 2

U_nonbonded_init=0.0
for i in range(n_atoms):
    for j in range(n_atoms):
        if i != j:
            U_nonbonded_init += 4* k_lj*( (d_lj/np.linalg.norm(y[i] - y[j]))**12 - (d_lj/np.linalg.norm(y[i] - y[j]))**6 )

U_init = U_angles_init + U_bonds_init + U_nonbonded_init
print("U init : " + str(U_init))

while(U_init>U_lim):

    # new direction
    x_md = np.random.normal(0, sigma_md, y_md.shape)
    y_tmp = x_md + y_md

    # Ep bonds
    U_bonds = 0
    for i in range(n_atoms-1) :
        r = np.linalg.norm(y_tmp[i]-y_tmp[i+1])
        U_bonds+= k_r*(r-r_md)**2

    # Ep angles
    U_angles = 0
    for i in range(n_atoms-2):
        theta = np.arccos(np.dot(y_tmp[i] - y_tmp[i+1], y_tmp[i+1] - y_tmp[i+2])
                          / (np.linalg.norm(y_tmp[i] - y_tmp[i+1]) * np.linalg.norm(y_tmp[i+1] - y_tmp[i+2])))
        U_angles += k_theta * (theta - theta_md) ** 2

    U_nonbonded = 0.0
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                U_nonbonded += 4 * k_lj * (
                            (d_lj / np.linalg.norm(y_tmp[i] - y_tmp[j])) ** 12 - (d_lj / np.linalg.norm(y_tmp[i] - y_tmp[j])) ** 6)

    # Ep total
    U_md= U_bonds+U_angles #+U_nonbonded
    if (U_init>U_md):
        U_init=U_md
        y_md=y_tmp
        print("yes : "+str(U_md))
        s_md+=np.var(x_md)
    # else:
    #     print("no : dU_bonds = "+str(U_bonds - U_bonds_init)+" ; dU_angles = "+str(U_angles- U_angles_init)+
    #           " ; dU_nonbonded = "+str(U_nonbonded- U_nonbonded_init)+" ; dU_md = "+str(U_md))

########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

sm = read_stan_model("md_nma", build=True)

model_dat = {'n_atoms': n_atoms,
             'n_modes':n_modes_fitted,
             'y':y_md,
             'x0':x,
             'A': A[:,7:(7+n_modes_fitted),:],
             'sigma':200,
             'epsilon':1,
             'mu':0,
             'U_init':U_lim,
             's_md':s_md,
             'k_r':k_r,
             'r0':r_md,
             'k_theta':k_theta,
             'theta0':theta_md,
             'k_lj':k_lj,
             'd_lj':d_lj
            }
fit = sm.sampling(data=model_dat, iter=1000, chains=4)
la = fit.extract(permuted=True)
q_res = la['q']
x_res = np.mean(la['x'], axis=0)

# fit using only NMA to compare
sm_nma = read_stan_model("nma_nmodes", build=False)
fit_nma = sm_nma.sampling(data=model_dat, iter=1000, chains=4)
la_nma = fit_nma.extract(permuted=True)
q_res_nma = la_nma['q']
x_res_nma = np.mean(la_nma['x'], axis=0)

#post process for visualization
x_res_full = la['x']
x_res_nma_full = la_nma['x']
n_iter = x_res_full.shape[0]
x_mse = np.zeros(n_iter)
x_mse_nma= np.zeros(n_iter)
for i in range(n_iter):
    x_mse[i] = np.sqrt(np.sum([np.linalg.norm(x_res_full[i,j] - y_md[j] )**2 for j in range(n_atoms)])/n_atoms)
    x_mse_nma[i] = np.sqrt(np.sum([np.linalg.norm(x_res_nma_full[i, j] - y_md[j])**2 for j in range(n_atoms)])/n_atoms)
########################################################################################################
#               RESULTS
########################################################################################################

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x[:,0],x[:,1], x[:,2], c='r')
# ax.scatter(y[:,0],y[:,1], y[:,2], c='b')
ax.scatter(y_md[:,0],y_md[:,1], y_md[:,2], c='r')
ax.scatter(x_res[:,0],x_res[:,1], x_res[:,2], c='g', marker="x", s=100)
ax.scatter(x_res_nma[:,0],x_res_nma[:,1], x_res_nma[:,2], c='b', marker="x", s=100)
ax.legend(["Ground Truth", "NMA + MD fit", "NMA fit"])
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
ax.set_xlabel("modes")
ax.set_ylabel("amplitude")
ax.legend(["Ground Truth"])
fig.suptitle("Normal modes amplitudes distribution")
fig.savefig("results/modes_amplitudes.png")
fig.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(x_mse,100,color= 'g')
ax.hist(x_mse_nma,100, color='b')
ax.legend(["NMA + MD fit", "NMA fit"])
fig.suptitle("RMSE")
fig.show()