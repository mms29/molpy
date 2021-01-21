import src.constants
import src.molecule
import numpy as np
from src.functions import *
from src.force_field import *


def run_md(mol, dt, total_time, temperature=300):

    n_steps = int(total_time/dt)
    xt = mol.coords
    vt = np.random.normal(0, np.sqrt(src.constants.K_BOLTZMANN*temperature / src.constants.CARBON_MASS), xt.shape)
    Ft = src.force_field.get_gradient(mol)

    molstep=[mol]
    Fstep=[Ft]

    for i in range(n_steps):
        print("ITER = "+str(i))

        # positions updates
        xt = xt + dt*vt + dt**2 *Ft/2

        # velocities update
        molt = src.molecule.Molecule.from_coords(xt)
        Ftt = src.force_field.get_gradient(molt)
        vt = vt + dt*(Ft +Ftt)/2
        Ft = Ftt

        molstep.append(molt)
        Fstep.append(Ftt)

    return molstep, Fstep

def MC_gradient_descend(init, target_density,n_iter,k, dt, size, sigma, sampling_rate, threshold):

    xt= init.coords
    l_U = []
    l_Up = []
    l_Ub = []
    for i in range(n_iter):

        psim = volume_from_pdb_fast3(coord=xt, size=size, sigma=sigma, sampling_rate=sampling_rate, threshold=threshold)
        Ub = get_RMSD(psim=psim, pexp=target_density.data)
        Up = get_energy(xt, verbose=False)
        U = k*Ub + Up

        dUb = get_grad_RMSD3(coord=xt, psim=psim, pexp=target_density.data, size=size, sampling_rate=sampling_rate, sigma=sigma, threshold=threshold)
        dUp = get_autograd(xt)
        dU = k*dUb + dUp
        F = -dU
        print("iter="+str(i)+" ; Ub="+str(k*Ub)+" ; Up="+str(Up)+" ; dUb="+str(np.mean(np.linalg.norm(dUb, axis=1) ))+" ; dUp="+str(np.mean(np.linalg.norm(dUp, axis=1) )))
        xt = xt + dt*F

        l_U.append(U)
        l_Up.append(Up)
        l_Ub.append(Ub)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(l_Ub)
    ax[0].set_title("RMSD Energy")
    ax[1].plot(l_Up)
    ax[1].set_title("Potential Energy")
    ax[2].plot(l_U)
    ax[2].set_title("Total Energy")

    return xt

def NMA_gradient_descend(init, target_density,n_iter, dt, size, sigma, sampling_rate, threshold):

    qt = np.zeros(init.modes.shape[1])
    l_dq=[]
    l_RMSD=[]

    for i in range(n_iter):

        coord = init.coords + np.dot(qt, init.modes)
        psim = volume_from_pdb_fast3(coord=coord, size=size, sigma=sigma, sampling_rate=sampling_rate, threshold=threshold)
        _, dq = get_grad_RMSD3_NMA(coord= coord, psim=psim, pexp=target_density.data, size=size, sampling_rate=sampling_rate,
                                   sigma=sigma, A_modes=init.modes, threshold=threshold)
        rmsd = get_RMSD(psim, target_density.data)

        print("iter="+str(i)+" ; rmsd="+str(rmsd)+" ; dq="+str(dq))
        qt = qt - dt*dq

        l_dq.append(np.linalg.norm(dq))
        l_RMSD.append(rmsd)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(l_RMSD)
    ax[0].set_title("RMSD ")
    ax[1].plot(l_dq)
    ax[1].set_title("Gradient Q")

    return init.coords + np.dot(qt, init.modes), qt

def MCNMA_gradient_descend(init, target_density,n_iter,k, dxt,dqt, size, sigma, sampling_rate, threshold):

    qt = np.zeros(init.modes.shape[1])
    xt = np.zeros(init.coords.shape)

    l_U = []
    l_Up = []
    l_Ub = []

    for i in range(n_iter):

        coord = init.coords + np.dot(qt, init.modes) + xt

        psim = volume_from_pdb_fast3(coord=coord, size=size, sigma=sigma, sampling_rate=sampling_rate, threshold=threshold)
        Ub = get_RMSD(psim=psim, pexp=target_density.data)
        Up = get_energy(coord, verbose=False)
        U = k*Ub + Up

        dUb, dq = get_grad_RMSD3_NMA(coord= coord, psim=psim, pexp=target_density.data, size=size, sampling_rate=sampling_rate,
                                   sigma=sigma, A_modes=init.modes, threshold=threshold)
        dUp = get_autograd(coord)
        dU = k*dUb + dUp
        F = -dU
        print("iter="+str(i)+" ; Ub="+str(k*Ub)+" ; Up="+str(Up)+" ; dUb="+str(np.mean(np.linalg.norm(dUb, axis=1) ))+" ; dUp="+str(np.mean(np.linalg.norm(dUp, axis=1) ))+" ; dq="+str(dq))

        qt += - dqt *dq
        xt += dxt*F

        l_U.append(U)
        l_Up.append(Up)
        l_Ub.append(Ub)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(l_Ub)
    ax[0].set_title("RMSD Energy")
    ax[1].plot(l_Up)
    ax[1].set_title("Potential Energy")
    ax[2].plot(l_U)
    ax[2].set_title("Total Energy")

    return coord, qt


def molecular_dynamics(init, target_density, n_iter,k, dt, size, sigma, sampling_rate, threshold):
    # Initial conditions
    xt = init.coords
    vt = np.random.normal(0, 1, xt.shape)
    # vt = np.random.normal(0, np.sqrt(src.constants.K_BOLTZMANN*Ti / (src.constants.CARBON_MASS* (3 * xt.shape[0]))), xt.shape)


    # Initial gradient
    psim = volume_from_pdb_fast3(coord=xt, size=size, sigma=sigma, sampling_rate=sampling_rate, threshold=threshold)
    dUb = get_grad_RMSD3(coord=xt, psim=psim, pexp=target_density.data, size=size, sampling_rate=sampling_rate,
                         sigma=sigma,
                         threshold=threshold)
    dUp = get_autograd(xt)
    dU = k * dUb + dUp
    F = -dU
    # T = K / (1 / 2 * src.constants.K_BOLTZMANN )

    l_Up = []
    l_Ub = []
    l_dUp = []
    l_dUb = []
    l_U = []
    l_dU = []
    l_F = []
    l_K = []
    l_T = []
    l_xt = []
    for i in range(n_iter):
        # Position update
        xt = xt + (dt * vt) + dt**2 * (F / 2)

        # Volume update
        psim = volume_from_pdb_fast3(coord=xt, size=size, sigma=sigma, sampling_rate=sampling_rate, threshold=threshold)

        # Potential energy update
        Ub = get_RMSD(psim=psim, pexp=target_density.data)
        Up = get_energy(xt, verbose=False)
        U = k * Ub + Up

        # Gradient Update
        dUb = get_grad_RMSD3(coord=xt, psim=psim, pexp=target_density.data, size=size, sampling_rate=sampling_rate,
                             sigma=sigma, threshold=threshold)
        dUp = get_autograd(xt)
        dU = k * dUb + dUp
        Ft = -dU

        # velocities update
        vt = vt + (dt * (F + Ft) / 2)
        F = Ft

        # Kinetic & temperature update
        K = src.force_field.get_kinetic_energy(vt)
        T = K / (3 / 2 * (3 * xt.shape[0]) * src.constants.K_BOLTZMANN)

        # velocities rescaling
        # vt = np.sqrt(Ti/T) * vt

        l_Up.append(Up)
        l_Ub.append(k * Ub)
        l_dUp.append((np.linalg.norm(dUp, axis=1)))
        l_dUb.append((np.linalg.norm(k * dUb, axis=1)))
        l_U.append(U)
        l_dU.append(dU)
        l_F.append(np.mean(np.linalg.norm(F, axis=1)))
        l_K.append(K)
        l_T.append(T)
        l_xt.append(xt)

        print("ITER=" + str(i) + " ; Ub=" + str(k * Ub) + " ; Up=" + str(Up) + " ; dUb=" + str(
            np.mean(np.linalg.norm(k * dUb, axis=1)))
              + " ; dUp=" + str(np.mean(np.linalg.norm(dUp, axis=1))) + " ; K=" + str(K) + " ; T=" + str(T))


    fig, ax =plt.subplots(2,3, figsize=(15,8))
    ax[0,0].plot(np.mean(np.array(l_dUp), axis=1))
    ax[0,0].set_title("Gradient Potential")
    ax[1,0].plot(np.mean(np.array(l_dUb), axis=1))
    ax[1,0].set_title("Gradient RMSD")
    ax[0,1].plot(l_Up)
    ax[0,1].set_title("Energy Potential")
    ax[1,1].plot(l_Ub)
    ax[1,1].set_title("Energy RMSD")
    ax[0,2].plot(l_U)
    ax[0,2].set_title("Energy Total")
    ax[1,2].plot(l_T)
    ax[1,2].set_title("Temperature")

    return xt
