import autograd.numpy as npg
import numpy as np
from autograd import grad, jacobian, elementwise_grad
import src.io
import matplotlib.pyplot as plt
import time
from src.functions import *
from mpl_toolkits.mplot3d import axes3d
import src.constants
import src.force_field
import src.simulation
import src.functions
import src.flexible_fitting
from src.viewers import structures_viewer

# import PDB
init =src.io.read_pdb("data/AK/AK.pdb")
init.add_modes("data/AK/modes/vec.", n_modes=4)
init = init.select_atoms(pattern='CA')


size,sampling_rate,gaussian_sigma = 32, 4, 2
sim = src.simulation.Simulator(init)
target = sim.nma_deform([168,0,-300,0])
# target = sim.mc_deform(v_bonds=0, v_angles=0, v_torsions=0.01)
# structures_viewer([target, init, nma])
target_density = target.to_density(size,sampling_rate=sampling_rate,gaussian_sigma=gaussian_sigma, threshold=5)
init_density = init.to_density(size,sampling_rate=sampling_rate,gaussian_sigma=gaussian_sigma, threshold=5)
# target_density.show()

def get_RMSD_NMA(coord, q_modes, pexp, size, sampling_rate, sigma, A_modes):
    rmsd = 0
    qAmodes = npg.dot(q_modes, A_modes)

    for i in range(size):
        for j in range(size):
            for k in range(size):
                mu = ((npg.array([i, j, k]) - npg.ones(3) * (size / 2)) * sampling_rate)
                x = coord + qAmodes
                psim = npg.sum(npg.exp(-npg.square(npg.linalg.norm(x  - mu, axis=1)) / (2 * (sigma ** 2))))
                rmsd += npg.square(psim-pexp[i,j,k])
    return rmsd

def get_grad_RMSD_NMA(coord, psim, pexp, size, sampling_rate, sigma, q_modes, A_modes):

    n_atoms = coord.shape[0]
    n_modes = q_modes.shape[0]
    pdiff = psim - pexp

    dx = np.zeros(coord.shape)
    dq = np.zeros(n_modes)
    mu = (np.mgrid[0:size, 0:size, 0:size] - size / 2) * sampling_rate
    for i in range(n_atoms):
        x = np.repeat(coord[i] + np.dot(q_modes, A_modes[i]), size ** 3).reshape(3, size, size, size)
        tmp = 2* pdiff*np.exp(-np.square(np.linalg.norm(x-mu, axis=0))/(2*(sigma ** 2)))

        dpsimx =-(1/(sigma**2)) * (x-mu) * np.array([tmp,tmp,tmp])
        dx[i] = np.sum(dpsimx, axis=(1,2,3))

        dpsimq = -(1 / (sigma ** 2)) * (x - mu) * np.array([tmp, tmp, tmp])
        dq += np.dot(A_modes[i] , np.sum(dpsimq, axis=(1, 2, 3)))

    return dx, dq

def get_grad_RMSD3_NMA(coord, psim, pexp, size, sampling_rate, sigma, q_modes, A_modes, threshold):
    vox, n_vox = select_voxels(coord, size, sampling_rate, threshold)
    n_atoms = coord.shape[0]
    n_modes = q_modes.shape[0]
    pdiff = psim - pexp

    dx = np.zeros(coord.shape)
    dq = np.zeros(n_modes)
    for i in range(n_atoms):
        mu = (np.mgrid[vox[i, 0]:vox[i, 0] + n_vox,
              vox[i, 1]:vox[i, 1] + n_vox,
              vox[i, 2]:vox[i, 2] + n_vox] - size / 2) * sampling_rate
        x = np.repeat(coord[i] + np.dot(q_modes, A_modes[i]), n_vox ** 3).reshape(3, n_vox, n_vox, n_vox)
        tmp = 2 * pdiff[vox[i, 0]:vox[i, 0] + n_vox,
                  vox[i, 1]:vox[i, 1] + n_vox,
                  vox[i, 2]:vox[i, 2] + n_vox] * np.exp(-np.square(np.linalg.norm(x - mu, axis=0)) / (2 * (sigma ** 2)))

        dpsimx =-(1/(sigma**2)) * (x-mu) * np.array([tmp,tmp,tmp])
        dx[i] = np.sum(dpsimx, axis=(1,2,3))

        dpsimq = -(1 / (sigma ** 2)) * (x - mu) * np.array([tmp, tmp, tmp])
        dq += np.dot(A_modes[i] , np.sum(dpsimq, axis=(1, 2, 3)))

    return dx, dq



q_modes =npg.array([0,0,0,0])

t = time.time()
dRMSD  = elementwise_grad(get_RMSD_NMA,1)
g1 = dRMSD(init.coords, q_modes,target_density.data, size, sampling_rate, gaussian_sigma,init.modes)
print("dt1="+str(time.time()-t))
print(g1)

t = time.time()
psim = volume_from_pdb_fast3(coord=init.coords + np.dot(q_modes, init.modes), size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=5)
_,g2 = get_grad_RMSD_NMA(coord=init.coords, psim =psim ,pexp=target_density.data, size=size,
                       sampling_rate=sampling_rate, sigma=gaussian_sigma, q_modes=q_modes, A_modes=init.modes)
print("dt2="+str(time.time()-t))
print(g2)

t = time.time()
psim = volume_from_pdb_fast3(coord=init.coords + np.dot(q_modes, init.modes), size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=5)
_,g3 = get_grad_RMSD3_NMA(coord=init.coords, psim =psim ,pexp=target_density.data, size=size,
                       sampling_rate=sampling_rate, sigma=gaussian_sigma, q_modes=q_modes, A_modes=init.modes, threshold=5)
print("dt3="+str(time.time()-t))
print(g3)

get_RMSD(psim, target_density.data)
get_RMSD_NMA(coord= init.coords,q_modes= q_modes, pexp=target_density.data, size=size, sampling_rate=sampling_rate,
             sigma=gaussian_sigma, A_modes=init.modes)


x_res, q_res = src.flexible_fitting.MCNMA_gradient_descend(init=init, target_density=target_density,n_iter=200, dxt=0.001,
                                    dqt=50, k = 100,
                                   size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=5)
x_res2, q_res2 = src.flexible_fitting.NMA_gradient_descend(init=init, target_density=target_density,n_iter=15, dt=50,
                                   size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=5)
x_res3 = src.flexible_fitting.MC_gradient_descend(init=init, target_density=target_density,n_iter=1000, dt=0.001, k=1000,
                                   size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=3)
plot_structure([target.coords, x_res2, init.coords], ["target", "res", "init"])
#########################################################################################################

import autograd.numpy as npg
import numpy as np
from autograd import grad, jacobian, elementwise_grad
import src.io
import matplotlib.pyplot as plt
import time
from src.functions import *
from mpl_toolkits.mplot3d import axes3d
import src.constants
import src.force_field
import src.simulation
import src.viewers
from src.force_field import get_energy, get_autograd
# import PDB
init =src.io.read_pdb("data/AK/AK.pdb")
init.add_modes("data/AK/modes/vec.", n_modes=3)
init.center_structure()
init = init.select_atoms(pattern='CA')

sim = src.simulation.Simulator(init)
target = sim.nma_deform( amplitude=[200,-100,0])
# target = sim.mc_deform(v_bonds=0.01, v_angles=0.001, v_torsions = 0.01)
plot_structure([init.coords,  target.coords], ["init", "target"])


size= 128
sampling_rate=4
sigma=2
threshold=3
target_density = target.to_density(size=size, sampling_rate=sampling_rate, gaussian_sigma=sigma, threshold=threshold)
target_density.show()


k=100
Ti = 0
dt=0.001
xt= init.coords
vt = np.random.normal(0,1, xt.shape)
#vt = np.random.normal(0, np.sqrt(src.constants.K_BOLTZMANN*Ti / (src.constants.CARBON_MASS* (3 * xt.shape[0]))), xt.shape)


psim = volume_from_pdb_fast3(coord=xt, size=size, sigma=sigma, sampling_rate=sampling_rate, threshold=threshold)
Ub = get_RMSD(psim=psim, pexp=target_density.data)
Up = get_energy(init, verbose=False)
U = k * Ub + Up

t = time.time()
dUb = get_grad_RMSD3_NMA(coord=xt, psim=psim, pexp=target_density.data, size=size, sampling_rate=sampling_rate, sigma=sigma,
                     threshold=threshold, A_modes=init.modes)
print(time.time() -t)
dUp = get_autograd(xt)
dU = k * dUb + dUp
F = -dU
K = src.force_field.get_kinetic_energy(vt)
T = K / (3 / 2 * (3*xt.shape[0])* src.constants.K_BOLTZMANN )
# T = K / (1 / 2 * src.constants.K_BOLTZMANN )

# vt = np.sqrt(Ti/T) * vt

# vt=vt_test


l_Up = []
l_Ub = []
l_dUp = []
l_dUb = []
l_U = []
l_dU = []
l_F = []
l_K = []
l_T = []
l_xt=[]
for i in range(1000):

    # Position update
    xt = xt + (dt*vt) + (dt**2)*(F/2)

    # Volume update
    psim = volume_from_pdb_fast3(coord=xt, size=size, sigma=sigma, sampling_rate=sampling_rate, threshold=threshold)

    # Potential energy update
    Ub = get_RMSD(psim=psim, pexp=target_density.data)
    Up = get_energy(xt, verbose=False)
    U = k*Ub + Up

    # Gradient Update
    dUb = get_grad_RMSD3(coord=xt, psim=psim, pexp=target_density.data, size=size, sampling_rate=sampling_rate, sigma=sigma, threshold=threshold)
    dUp = get_autograd(xt)
    dU = k*dUb + dUp
    Ft = -dU

    # velocities update
    vt = vt + (dt * (F + Ft) / 2)
    F = Ft

    # Kinetic & temperature update
    K = src.force_field.get_kinetic_energy(vt)
    T = K / (3 / 2 * (3*xt.shape[0])* src.constants.K_BOLTZMANN )

    # velocities rescaling
    # vt = np.sqrt(Ti/T) * vt

    l_Up.append(Up)
    l_Ub.append(k*Ub)
    l_dUp.append((np.linalg.norm(dUp, axis=1) ))
    l_dUb.append((np.linalg.norm(k*dUb, axis=1) ))
    l_U.append(U)
    l_dU.append(dU)
    l_F.append(np.mean(np.linalg.norm(F, axis=1) ))
    l_K.append(K)
    l_T.append(T)
    l_xt.append(xt)

    print("ITER="+str(i)+" ; Ub="+str(k*Ub)+" ; Up="+str(Up)+" ; dUb="+str(np.mean(np.linalg.norm(k*dUb, axis=1) ))
          +" ; dUp="+str(np.mean(np.linalg.norm(dUp, axis=1) ))+" ; K="+str(K)+" ; T="+str(T))

plot_structure([l_xt[-1], target.coords], [ "fit", "target"])
print(get_energy(l_xt[-1]))

fig, ax =plt.subplots(2,2)
ax[0,0].plot(np.mean(np.array(l_dUp), axis=1))
ax[0,0].set_title("Gradient Potential")
ax[0,1].plot(np.mean(np.array(l_dUb), axis=1))
ax[0,1].set_title("Gradient RMSD")
ax[1,0].plot(l_Up)
ax[1,0].set_title("Energy Potential")
ax[1,1].plot(l_Ub)
ax[1,1].set_title("Energy RMSD")

