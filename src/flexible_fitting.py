import src.constants
import src.molecule
import numpy as np
from src.functions import *
from src.force_field import *
import time
import pickle


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

        molt = src.molecule.Molecule(xt, chain_id=init.chain_id)
        psim = volume_from_pdb_fast3(coord=xt, size=size, sigma=sigma, sampling_rate=sampling_rate, threshold=threshold)
        Ub = get_RMSD(psim=psim, pexp=target_density.data)
        Up = get_energy(molt, verbose=False)
        U = k*Ub + Up

        dUb = get_grad_RMSD3(coord=xt, psim=psim, pexp=target_density.data, size=size, sampling_rate=sampling_rate, sigma=sigma, threshold=threshold)
        dUp = get_autograd(molt)
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

def NMA_gradient_descend(init, target_density,n_iter, dt, size, sigma, sampling_rate, threshold, q_init=None):

    if q_init is not None:
        qt = q_init
    else:
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

def MCNMA_gradient_descend(init, target_density,n_iter,k, dxt,dqt, size, sigma, sampling_rate, threshold, q_init=None):
    if q_init is not None:
        qt = q_init
    else:
        qt = np.zeros(init.modes.shape[1])
    xt = np.zeros(init.coords.shape)

    l_U = []
    l_Up = []
    l_Ub = []

    for i in range(n_iter):

        coord = init.coords + np.dot(qt, init.modes) + xt
        molt = src.molecule.Molecule(coord, chain_id=init.chain_id)

        psim = volume_from_pdb_fast3(coord=coord, size=size, sigma=sigma, sampling_rate=sampling_rate, threshold=threshold)
        Ub = get_RMSD(psim=psim, pexp=target_density.data)
        Up = get_energy(molt, verbose=False)
        U = k*Ub + Up

        dUb, dq = get_grad_RMSD3_NMA(coord= coord, psim=psim, pexp=target_density.data, size=size, sampling_rate=sampling_rate,
                                   sigma=sigma, A_modes=init.modes, threshold=threshold)
        dUp = get_autograd(molt)
        dU = k*dUb + dUp
        F = -dU
        print("iter="+str(i)+" ; Ub="+str(k*Ub)+" ; Up="+str(Up)+" ; dUb="+str(np.mean(np.linalg.norm(dUb, axis=1) ))+" ; dUp="+str(np.mean(np.linalg.norm(dUp, axis=1) ))+" ; dq="+str(dq))

        qt = qt - dqt *dq
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
    dUp = get_autograd(init)
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
    l_criterion = []
    for i in range(n_iter):
        # Position update
        xt = xt + (dt * vt) + dt**2 * (F / 2)
        molt = src.molecule.Molecule(xt, chain_id=init.chain_id)

        # Volume update
        psim = volume_from_pdb_fast3(coord=xt, size=size, sigma=sigma, sampling_rate=sampling_rate, threshold=threshold)

        # Potential energy update
        Ub = get_RMSD(psim=psim, pexp=target_density.data)
        Up = get_energy(molt, verbose=False)
        U = k * Ub + Up

        # Gradient Update
        dUb = get_grad_RMSD3(coord=xt, psim=psim, pexp=target_density.data, size=size, sampling_rate=sampling_rate,
                             sigma=sigma, threshold=threshold)
        dUp = get_autograd(molt)
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

        criterion = np.dot((xt.flatten() - init.coords.flatten()), vt.flatten())
        l_criterion.append(criterion)

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
    ax[1,2].plot(l_criterion)
    ax[1,2].set_title("Criterion")

    return xt


def HMC(init, target_density, n_iter, n_warmup,k, dt, max_iter):
    molt = init

    l_Up = []
    l_Ub = []
    l_U = []
    l_L = []
    l_a =[]
    l_xt =[]
    l_c = []
    l_cc =[]
    ll_xt=[]
    t = time.time()
    for i in range(n_iter):
        print("HMC ITER = "+str(i))
        xt, Up, Ub, U, a, L , c, lxt, cc= HMC_step(init=molt, target_density=target_density,
                    k=k, dt=dt, max_iter=max_iter)
        molt = src.molecule.Molecule(xt, chain_id=init.chain_id)
        l_Up += Up
        l_Ub += Ub
        l_U +=U
        l_c += c
        l_L.append(L)
        l_a.append(a)
        l_xt.append(xt)
        ll_xt += lxt
        l_cc += cc

    fig, ax =plt.subplots(2,3, figsize=(15,8))
    ax[0,0].plot(l_Up)
    ax[0,0].set_title("Energy Potential")
    ax[1,0].plot(l_Ub)
    ax[1,0].set_title("Energy RMSD")
    ax[0,1].plot(l_U)
    ax[0,1].set_title("Energy Total")
    ax[1,1].plot(np.convolve(np.array(l_a), np.ones(5), 'valid') / 5)
    ax[1,1].set_title("Acceptance rate")
    ax[0,2].plot(l_L)
    ax[0,2].set_title("Transition length")
    ax[1, 2].plot(l_c)
    ax[1, 2].set_title("Criterion")
    for i in range(len(l_L)):
        ax[0,0].axvline(np.sum(l_L[:i+1]))
        ax[1,0].axvline(np.sum(l_L[:i+1]))
        ax[0,1].axvline(np.sum(l_L[:i+1]))
        ax[1,2].axvline(np.sum(l_L[:i+1]))

    print("EXECUTION ENDED : t="+str(time.time()-t)+"  iter="+str(np.sum(l_L)) +"  t/iter="+str((time.time()-t)/np.sum(l_L)))
    mol = src.molecule.Molecule(coords = np.mean(l_xt[n_warmup:], axis=0), chain_id=init.chain_id)
    mol.get_energy()
    mol_density = mol.to_density(size=target_density.size, sampling_rate=target_density.sampling_rate,
                                 gaussian_sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
    cc= cross_correlation(mol_density.data, target_density.data)
    print("CC = "+str(cc))


    return mol, ll_xt, l_L, l_cc

def HMC_step(init, target_density,k, dt, max_iter):
    t=time.time()

    # Initial conditions
    xt = init.coords
    vt = np.random.normal(0, 1, xt.shape)
    # vt = np.random.normal(0, np.sqrt(src.constants.K_BOLTZMANN*Ti / (src.constants.CARBON_MASS* (3 * xt.shape[0]))), xt.shape)

    # Initial gradient
    psim = volume_from_pdb_fast3(coord=xt, size=target_density.size, sampling_rate=target_density.sampling_rate,
                                 sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
    Ub = get_RMSD(psim=psim, pexp=target_density.data)
    Up = get_energy(init, verbose=False)
    U = k * Ub + Up
    dUb = get_grad_RMSD3(coord=xt, psim=psim, pexp=target_density.data,size=target_density.size, sampling_rate=target_density.sampling_rate,
                                 sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
    dUp = get_autograd(init)
    dU = k * dUb + dUp
    F = -dU
    K = src.force_field.get_kinetic_energy(vt)
    criterion = np.dot((xt.flatten() - init.coords.flatten()), vt.flatten())
    i=0

    H_init = U + K

    l_Up = [Up]
    l_Ub = [Ub]
    l_U = [U]
    l_c =[]
    l_cc=[]
    l_xt =[]
    print("dt="+str(time.time()-t))

    while(i<max_iter):


        # Position update
        xt = xt + (dt * vt) + dt ** 2 * (F / 2)
        molt = src.molecule.Molecule(xt, chain_id=init.chain_id)

        # Volume update
        psim = volume_from_pdb_fast3(coord=xt, size=target_density.size, sampling_rate=target_density.sampling_rate,
                                     sigma=target_density.gaussian_sigma, threshold=target_density.threshold)

        # Potential energy update
        Ub = get_RMSD(psim=psim, pexp=target_density.data)
        Up = get_energy(molt, verbose=False)
        U = k * Ub + Up

        # Gradient Update
        dUb = get_grad_RMSD3(coord=xt, psim=psim, pexp=target_density.data, size=target_density.size,
                             sampling_rate=target_density.sampling_rate,
                             sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
        dUp = get_autograd(molt)
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

        criterion = np.dot((xt.flatten() - init.coords.flatten()), vt.flatten())
        i+=1

        print("ITER=" + str(i) + " ; Ub=" + str(k * Ub) + " ; Up=" + str(Up) + " ; dUb=" + str(
            np.mean(np.linalg.norm(k * dUb, axis=1)))
              + " ; dUp=" + str(np.mean(np.linalg.norm(dUp, axis=1))) + " ; K=" + str(K) + " ; Crit=" + str(criterion))

        l_Up.append(Up)
        l_Ub.append(Ub)
        l_U .append(U)
        l_c.append(criterion)
        l_xt.append(xt)
        l_cc.append(cross_correlation(psim, target_density.data))

    H = U+K

    accept_p = np.min([1 , H_init/H])
    if int(np.random.choice(2, p=[1-accept_p, accept_p])) == 0:
        xt = init.coords
        print("REJECTED")
        a=0
    else:
        print("ACCEPTED")
        a=1

    return xt, l_Up, l_Ub, l_U, a, i, l_c, l_xt, l_cc
















def HMCNMA(init, target_density, n_iter, n_warmup, max_iter, k, dxt, dqt, m_test,q_init=None):
    xt = np.zeros(init.coords.shape)
    if q_init is not None:
        qt = q_init
    else:
        qt = np.zeros(init.modes.shape[1])

    l_Up = []
    l_Ub = []
    l_U = []
    l_L = []
    l_a =[]
    l_xt =[]
    l_qt=[]
    l_c = []
    l_c2=[]
    ll_xt=[]
    ll_qt=[]
    l_cc=[]
    t=time.time()
    for i in range(n_iter):
        print("HMC ITER = "+str(i))
        xt, qt,  Up, Ub, U, c2, L , c, lxt, lqt, cc= HMCNMA_step(init=init, x_init=xt, q_init = qt, target_density=target_density,
                    k=k, dxt=dxt, dqt=dqt, max_iter=max_iter, m_test=m_test)

        l_Up += Up
        l_Ub += Ub
        l_U +=U
        l_c += c
        l_L.append(L)
        l_c2 += c2
        l_xt.append(xt)
        l_qt.append(qt)
        ll_xt += lxt
        ll_qt += lqt
        l_cc += cc

    fig, ax =plt.subplots(2,3, figsize=(15,8))
    ax[0,0].plot(l_Up)
    ax[0,0].set_title("Energy Potential")
    ax[1,0].plot(l_Ub)
    ax[1,0].set_title("Energy RMSD")
    ax[0,1].plot(l_U)
    ax[0,1].set_title("Energy Total")
    ax[1,1].plot(l_c2) #np.convolve(np.array(l_a), np.ones(5), 'valid') / 5
    ax[1,1].set_title("Acceptance rate")
    ax[0,2].plot(l_L)
    ax[0,2].set_title("Transition length")
    ax[1, 2].plot(l_c)
    ax[1, 2].set_title("Criterion")
    for i in range(len(l_L)):
        ax[0,0].axvline(np.sum(l_L[:i+1]))
        ax[1,0].axvline(np.sum(l_L[:i+1]))
        ax[0,1].axvline(np.sum(l_L[:i+1]))
        ax[1,2].axvline(np.sum(l_L[:i+1]))
        ax[1,1].axvline(np.sum(l_L[:i+1]))

    print("EXECUTION ENDED : t="+str(time.time()-t)+"  iter="+str(np.sum(l_L)) +"  t/iter="+str((time.time()-t)/np.sum(l_L)))
    coord = init.coords + np.mean(l_xt[n_warmup:], axis=0) + np.dot(np.mean(l_qt[n_warmup:], axis=0), init.modes)
    mol = src.molecule.Molecule(coords = coord, chain_id=init.chain_id)
    mol.get_energy()
    mol_density = mol.to_density(size=target_density.size, sampling_rate=target_density.sampling_rate,
                                 gaussian_sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
    cc= cross_correlation(mol_density.data, target_density.data)
    print("CC = "+str(cc))


    return mol, ll_xt, ll_qt, l_L, l_cc

def HMCNMA_step(init, x_init, q_init,  target_density, k, dxt, dqt, max_iter,m_test):
    # Initial conditions

    xt = x_init
    qt = q_init
    vt = np.random.normal(0, 1, xt.shape)
    vqt = np.random.normal(0, m_test, qt.shape)
    init_coord = init.coords + np.dot(qt, init.modes) + xt
    coordt= init_coord
    molt = src.molecule.Molecule(coordt, modes=init.modes, chain_id=init.chain_id)

    # Initial Energy
    psim = volume_from_pdb_fast3(coord=coordt, size=target_density.size, sampling_rate=target_density.sampling_rate,
                                 sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
    Ub = get_RMSD(psim=psim, pexp=target_density.data)
    Up = get_energy(molt, verbose=False)
    U = k * Ub + Up

    #Initial gradient
    dUb, dUq = get_grad_RMSD3_NMA(coord=coordt, psim=psim, pexp=target_density.data, size=target_density.size, A_modes=molt.modes,
            sampling_rate=target_density.sampling_rate, sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
    dUp, dUpq = get_autograd_NMA(init, xt, qt, init.modes)
    F = -(k * dUb + dUp)
    Fq = - (k * dUq + dUpq)
    K = get_kinetic_energy(vt)
    criterion = 0

    i=0
    print("ITER=" + str(i) + " ; Ub=" + str(k * Ub) + " ; Up=" + str(Up) + " ; dUq=" + str(k * dUq)
          + " ; dUpq=" + str(dUpq) + " ; q=" + str(qt) + " ; K=" + str(K) + " ; Crit=" + str(criterion))

    H_init = U + K

    l_Up = [Up]
    l_Ub = [Ub]
    l_U = [U]
    l_c =[]
    l_c2=[]
    l_xt=[]
    l_qt=[]
    l_cc=[]

    while( i< max_iter):


        # Position update
        xt = xt + (dxt * vt) + dxt ** 2 * (F / 2)
        qt = qt + (dqt * vqt) + dqt ** 2 * (Fq / 2)

        coordt = init.coords + np.dot(qt, molt.modes) + xt
        molt = src.molecule.Molecule(coordt, chain_id=molt.chain_id, modes=molt.modes)

        # Volume update
        psim = volume_from_pdb_fast3(coord=coordt, size=target_density.size, sampling_rate=target_density.sampling_rate,
                                     sigma=target_density.gaussian_sigma, threshold=target_density.threshold)

        # Potential energy update
        Ub = get_RMSD(psim=psim, pexp=target_density.data)
        Up = get_energy(molt, verbose=False)
        U = k * Ub + Up

        # Gradient Update
        dUb, dUq = get_grad_RMSD3_NMA(coord=coordt, psim=psim, pexp=target_density.data, size=target_density.size,
                                      A_modes=molt.modes,
                                      sampling_rate=target_density.sampling_rate, sigma=target_density.gaussian_sigma,
                                      threshold=target_density.threshold)
        dUp, dUpq = get_autograd_NMA(init, xt, qt, init.modes)
        Ft = -(k * dUb + dUp)
        Fqt = - (k * dUq + dUpq)

        # velocities update
        vt = vt + (dxt * (F + Ft) / 2)
        F = Ft
        vqt = vqt + (dqt * (Fq + Fqt) / 2)
        Fq = Fqt

        # Kinetic & temperature update
        K = get_kinetic_energy(vt)
        T = K / (3 / 2 * (3 * xt.shape[0]) * src.constants.K_BOLTZMANN)

        # velocities rescaling
        # vt = np.sqrt(Ti/T) * vt
        c2 = np.dot((qt - q_init), vqt)
        c1 =  np.dot((coordt.flatten() - init_coord.flatten()), vt.flatten())
        criterion = c1 + c2
        i+=1

        print("ITER=" + str(i) + " ; Ub=" + str(k * Ub) + " ; Up=" + str(Up) + " ; dUq=" + str(k*dUq)
              + " ; dUpq=" + str(dUpq) + " ; q=" + str(qt) + " ; K=" + str(K) + " ; Crit=" + str(criterion))

        l_Up.append(Up)
        l_Ub.append(Ub)
        l_U .append(U)
        l_c.append(c1)
        l_c2.append(c2)
        l_xt.append(xt)
        l_qt.append(qt)
        l_cc.append(cross_correlation(psim, target_density.data))

    H = U+K

    accept_p = np.min([1 , H_init/H])
    if int(np.random.choice(2, p=[1-accept_p, accept_p])) == 0:
        xt = x_init
        qt = q_init
        print("REJECTED")
        a=0
    else:
        print("ACCEPTED")
        a=1

    return xt, qt, l_Up, l_Ub, l_U, l_c2, i, l_c, l_xt, l_qt, l_cc


def HMCNMA2(init, target_density, n_iter, n_warmup, max_iter, lambda1,lambda2,lambda3, lambda4, dxt, dqt, q_init=None):
    xt = np.zeros(init.coords.shape)
    if q_init is not None:
        qt = q_init
    else:
        qt = np.zeros(init.modes.shape[1])

    # resuslts
    l_xt = []
    l_qt = []

    # energy
    l_U_potential = []
    l_U_biased = []
    l_U_modes = []
    l_U_positions = []
    l_U = []
    l_K = []

    # infos
    l_L = []
    l_c = []

    t=time.time()
    for i in range(n_iter):
        print("HMC ITER = "+str(i))
        xt, qt, L, c, U_potential, U_biased, U_modes, U_positions, U, K = HMCNMA2_step(init=init, x_init=xt, q_init = qt, target_density=target_density,
                    lambda1=lambda1,lambda2=lambda2,lambda3=lambda3, lambda4=lambda4, dxt=dxt, dqt=dqt, max_iter=max_iter)
        # resuslts
        l_xt.append(xt)
        l_qt.append(qt)

        # energy
        l_U_potential += U_potential
        l_U_biased += U_biased
        l_U_modes += U_modes
        l_U_positions += U_positions
        l_U += U
        l_K += K

        # infos
        l_L.append(L)
        l_c += c


    fig, ax =plt.subplots(2,3, figsize=(15,8))
    ax[0,0].plot(l_U_potential)
    ax[0,0].set_title("U_potential")
    ax[1,0].plot(l_U_biased)
    ax[1,0].set_title("U_bisaed")
    ax[0,1].plot(l_U_modes)
    ax[0,1].set_title("U_modes")
    ax[1,1].plot(l_U_positions)
    ax[1,1].set_title("U_positions")
    ax[0,2].plot(l_U)
    ax[0,2].plot(l_K)
    ax[0,2].plot(np.array(l_K)+np.array(l_U))
    ax[0,2].set_title("U_total")
    ax[1, 2].plot(l_c)
    ax[1, 2].set_title("Criterion")
    for i in range(len(l_L)):
        ax[0,0].axvline(np.sum(l_L[:i+1]))
        ax[1,0].axvline(np.sum(l_L[:i+1]))
        ax[0,1].axvline(np.sum(l_L[:i+1]))
        ax[1,1].axvline(np.sum(l_L[:i+1]))
        ax[0,2].axvline(np.sum(l_L[:i+1]))
        ax[1,2].axvline(np.sum(l_L[:i+1]))

    print("EXECUTION ENDED : t="+str(time.time()-t)+"  iter="+str(np.sum(l_L)) +"  t/iter="+str((time.time()-t)/np.sum(l_L)))
    coord = init.coords + np.mean(l_xt[n_warmup:], axis=0) + np.dot(np.mean(l_qt[n_warmup:], axis=0), init.modes)
    mol = src.molecule.Molecule(coords = coord, chain_id=init.chain_id)
    mol.get_energy()
    mol_density = mol.to_density(size=target_density.size, sampling_rate=target_density.sampling_rate,
                                 gaussian_sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
    cc= cross_correlation(mol_density.data, target_density.data)
    print("CC = "+str(cc))


    return mol, np.array(l_xt), np.array(l_qt)

def HMCNMA2_step(init, x_init, q_init,  target_density, lambda1,lambda2,lambda3, lambda4, dxt, dqt, max_iter):
    # Initial conditions

    xt = x_init
    vt = np.random.normal(0, 1, xt.shape)
    qt = q_init
    vqt = np.random.normal(0, 1, qt.shape)
    init_coord = init.coords + np.dot(qt, init.modes) + xt
    coordt= init_coord
    molt = src.molecule.Molecule(coordt, modes=init.modes, chain_id=init.chain_id)

    # Initial Energy
    psim = volume_from_pdb_fast3(coord=coordt, size=target_density.size, sampling_rate=target_density.sampling_rate,
                                 sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
    U_biased = get_RMSD(psim=psim, pexp=target_density.data)
    U_potential = get_energy(molt, verbose=False)
    U_modes = np.square(np.linalg.norm(qt))
    U_positions = np.square(np.linalg.norm(xt))
    U = (lambda1 * U_biased) + (lambda2 * U_potential) + (lambda3 * U_modes) + (lambda4 * U_positions)

    #Initial gradient
    dU_biased_x, dU_biased_q = get_grad_RMSD3_NMA(coord=coordt, psim=psim, pexp=target_density.data, size=target_density.size, A_modes=molt.modes,
            sampling_rate=target_density.sampling_rate, sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
    dU_potential_x, dU_potential_q = get_autograd_NMA(init, xt, qt, init.modes)
    dU_modes = 2*qt
    dU_positions = 2*xt
    Fx = -((lambda1 * dU_biased_x) + (lambda2 * dU_potential_x) + (lambda4 * dU_positions))
    Fq = -((lambda1 * dU_biased_q) + (lambda2 * dU_potential_q) + (lambda3 * dU_modes))
    K =  1/2 * np.sum(np.square(vt)) + 1/2 * np.sum(np.square(vqt))
    criterion = 0

    i=0
    H_init = U + K

    l_c =[]
    l_U_potential =[]
    l_U_biased =[]
    l_U_modes =[]
    l_U_positions =[]
    l_U =[]
    l_K =[]

    while(criterion >= 0 and i< max_iter):


        # Position update
        xt = xt + (dxt * vt) + dxt ** 2 * (Fx / 2)
        qt = qt + (dqt * vqt) + dqt ** 2 * (Fq / 2)

        coordt = init.coords + np.dot(qt, molt.modes) + xt
        molt = src.molecule.Molecule(coordt, chain_id=molt.chain_id, modes=molt.modes)

        # Volume update
        psim = volume_from_pdb_fast3(coord=coordt, size=target_density.size, sampling_rate=target_density.sampling_rate,
                                     sigma=target_density.gaussian_sigma, threshold=target_density.threshold)

        # Potential energy update
        U_biased = get_RMSD(psim=psim, pexp=target_density.data)
        U_potential = get_energy(molt, verbose=False)
        U_modes = np.square(np.linalg.norm(qt))
        U_positions = np.square(np.linalg.norm(xt))
        U = (lambda1 * U_biased) + (lambda2 * U_potential) + (lambda3 * U_modes) + (lambda4 * U_positions)

        # Gradient Update
        dU_biased_x, dU_biased_q = get_grad_RMSD3_NMA(coord=coordt, psim=psim, pexp=target_density.data,
                                                      size=target_density.size, A_modes=molt.modes,
                                                      sampling_rate=target_density.sampling_rate,
                                                      sigma=target_density.gaussian_sigma,
                                                      threshold=target_density.threshold)
        dU_potential_x, dU_potential_q = get_autograd_NMA(init, xt, qt, init.modes)
        dU_modes = 2 * qt
        dU_positions = 2 * xt
        Fxt = -((lambda1 * dU_biased_x) + (lambda2 * dU_potential_x) + (lambda4 * dU_positions))
        Fqt = -((lambda1 * dU_biased_q) + (lambda2 * dU_potential_q) + (lambda3 * dU_modes))

        # velocities update
        vt = vt + (dxt * (Fx + Fxt) / 2)
        Fx = Fxt
        vqt = vqt + (dqt * (Fq + Fqt) / 2)
        Fq = Fqt

        # Kinetic & temperature update
        K = 1 / 2 * np.sum(np.square(vt)) + 1 / 2 * np.sum(np.square(vqt))

        # criterion update
        criterion = np.dot((coordt.flatten() - init_coord.flatten()), vt.flatten()) +np.dot((qt - q_init), vqt)

        print_step(["ITER", "U_biased", "U_potential", "U_modes", "U_postions", "qt", "dU_modes", "K", "crit"],
                   [i, U_biased * lambda1, U_potential*lambda2, U_modes*lambda3, U_positions*lambda4, qt,dU_modes,K, criterion])

        l_c.append(criterion)
        l_U_potential.append(U_potential*lambda1)
        l_U_biased.append(U_biased*lambda2)
        l_U_modes.append(U_modes*lambda3)
        l_U_positions.append(U_positions*lambda4)
        l_U.append(U)
        l_K.append(K)

        i+=1


    H = U+K

    accept_p = np.min([1 , np.exp(H_init-H)])
    if int(np.random.choice(2, p=[1-accept_p, accept_p])) == 0:
        xt = x_init
        qt = q_init
        print("REJECTED "+str(accept_p))
    else:
        print("ACCEPTED "+str(accept_p))

    return xt, qt, i, l_c, l_U_potential, l_U_biased, l_U_modes, l_U_positions, l_U, l_K











def HNMA(init, target_density, n_iter, n_warmup, k, dt, max_iter, q_init=None):
    if q_init is not None:
        qt = q_init
    else:
        qt = np.zeros(init.modes.shape[1])


    l_U = []
    l_L = []
    l_qt=[]
    l_c = []
    ll_qt =[]
    l_cc=[]
    t=time.time()
    for i in range(n_iter):
        print("HMC ITER = "+str(i))
        qt, U, L , c, lqt, cc= HNMA_step(init = init, q_init = qt, target_density=target_density, k=k,
                    dt=dt, max_iter=max_iter)

        l_U +=U
        l_c += c
        l_L.append(L)
        l_qt.append(qt)
        ll_qt += lqt
        l_cc += cc

    fig, ax =plt.subplots(1,3)
    ax[0].plot(l_U)
    ax[0].set_title("Energy Total")
    ax[1].plot(l_L)
    ax[1].set_title("Transition length")
    ax[2].plot(l_c)
    ax[2].set_title("Criterion")
    for i in range(len(l_L)):
        ax[0].axvline(np.sum(l_L[:i+1]))
        ax[2].axvline(np.sum(l_L[:i+1]))

    print("EXECUTION ENDED : t="+str(time.time()-t)+"  iter="+str(np.sum(l_L)) +"  t/iter="+str((time.time()-t)/np.sum(l_L)))
    coord = init.coords + np.dot(np.mean(l_qt[n_warmup:], axis=0), init.modes)
    mol = src.molecule.Molecule(coords = coord, chain_id=init.chain_id)
    mol.get_energy()
    mol_density = mol.to_density(size=target_density.size, sampling_rate=target_density.sampling_rate,
                                 gaussian_sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
    cc= cross_correlation(mol_density.data, target_density.data)
    print("CC = "+str(cc))

    return mol, ll_qt,  l_L, l_cc

def HNMA_step(init, q_init,  target_density,k,  dt, max_iter):
    # Initial conditions

    xt = q_init
    vt = np.random.normal(0, 1, q_init.shape)
    coordt = init.coords + np.dot(xt, init.modes)

    # Initial Energy
    psim = volume_from_pdb_fast3(coord=coordt, size=target_density.size, sampling_rate=target_density.sampling_rate,
                                 sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
    U = get_RMSD(psim=psim, pexp=target_density.data)

    #Initial gradient
    _, dU = get_grad_RMSD3_NMA(coord=coordt, psim=psim, pexp=target_density.data, size=target_density.size,
                                sampling_rate=target_density.sampling_rate, sigma=target_density.gaussian_sigma,
                                threshold=target_density.threshold, A_modes=init.modes)
    F = -k*dU
    K = get_kinetic_energy(vt)
    criterion = np.dot((xt - q_init), vt)

    i=0

    H_init = U + K

    l_U = []
    l_c =[]
    l_qt=[]
    l_cc=[]

    while(i <max_iter):


        # Position update
        xt = xt + (dt * vt) + dt ** 2 * (F / 2)

        coordt = init.coords + np.dot(xt, init.modes)
        molt = src.molecule.Molecule(coordt, chain_id=init.chain_id, modes=init.modes)

        # Volume update
        psim = volume_from_pdb_fast3(coord=coordt, size=target_density.size, sampling_rate=target_density.sampling_rate,
                                     sigma=target_density.gaussian_sigma, threshold=target_density.threshold)

        # Potential energy update
        U = get_RMSD(psim=psim, pexp=target_density.data)
        Up = get_energy(molt, verbose=False)

        # Gradient Update
        _, dU = get_grad_RMSD3_NMA(coord=coordt, psim=psim, pexp=target_density.data, size=target_density.size,
                                      sampling_rate=target_density.sampling_rate, sigma=target_density.gaussian_sigma,
                                      threshold=target_density.threshold, A_modes=init.modes)
        Ft = -k*dU

        # velocities update
        vt = vt + (dt * (F + Ft) / 2)
        F = Ft


        # Kinetic & temperature update
        K = get_kinetic_energy(vt)
        T = K / (3 / 2 * (3 * xt.shape[0]) * src.constants.K_BOLTZMANN)

        # velocities rescaling
        # vt = np.sqrt(Ti/T) * vt

        criterion =  np.dot((xt - q_init), vt)
        i+=1

        print("iter=" + str(i) + " ; rmsd=" + str(U) +"  ; UP=" +str(Up)+" ; q=" + str(xt)+ " ; Crit=" + str(criterion))

        l_U .append(U)
        l_c.append(criterion)
        l_qt.append(xt)
        l_cc.append(cross_correlation(psim, target_density.data))

    H = U+K

    accept_p = np.min([1 , H_init/H])
    if int(np.random.choice(2, p=[1-accept_p, accept_p])) == 0:
        coordt = init.coords
        xt = q_init
        print("REJECTED")
        a=0
    else:
        print("ACCEPTED")
        a=1

    return  xt, l_U, i, l_c, l_qt, l_cc


class FlexibleFitting:
    def __init__(self, init, target):
        self.init = init
        self.target = target
        self.fit = {}


    def HMC(self, mode, n_iter, n_warmup, params, verbose=False):

        # define variables to be estimated
        self.fit= {
            "U" : [],
            "U_potential": [],
            "U_biased" :[],
            "K" : [],
            "C" : [],
            "mol": [self.init],
            "molt": [],
            "L" : [],
            "CC" : []
        }
        if "HMC" in mode:
            self._set("x",[params["x_init"]])
            self._set("xt", [])
            self._set("vt", [])
        if "NMA" in mode:
            self._set("q" ,[params["q_init"]])
            self._set("qt",[])
            self._set("wt",[])
        if "ROT" in mode:
            self._set("angles" ,[params["angles_init"]])
            self._set("anglest",[])
            self._set("vanglest",[])

        # HMC Loop
        for i in range(n_iter):
            print("HMC ITER = " + str(i))
            self.HMC_step(mode, params, verbose=verbose)

        self.res = {
            "mol" : src.molecule.Molecule(np.mean(np.array([i.coords for i in self.fit["mol"][n_warmup+1:]]), axis=0), chain_id=self.init.chain_id)
        }
        if "HMC" in mode:
            self.res["x"] = np.mean(np.array(self.fit["x"][n_warmup+1:]), axis=0)
        if "NMA" in mode:
            self.res["q"] = np.mean(np.array(self.fit["q"][n_warmup+1:]), axis=0)
        if "ROT" in mode:
            self.res["angles"] = np.mean(np.array(self.fit["angles"][n_warmup + 1:]), axis=0)

        return self.res


    def _get(self, x):
        if isinstance(self.fit[x], list):
            return self.fit[x][-1]
        else:
            return self.fit[x]

    def _add(self, param, x):
        if param in self.fit:
            self.fit[param].append(x)
        else:
            self.fit[param] = x

    def _set(self, param, x):
        self.fit[param] = x

    def _get_energy(self, mode, params, psim):
        U = 0

        U_biased = get_RMSD(psim=psim, pexp=self.target.data) * params["lb"]
        self._add("U_biased", U_biased)
        U+= U_biased

        U_potential = get_energy(self._get("molt"), verbose=False)* params["lp"]
        self._add("U_potential", U_potential)

        if "HMC" in mode:
            U_positions = np.square(np.linalg.norm(self._get("xt"))) * params["lx"]
            U += U_positions
            U+= U_potential

        if "NMA" in mode:
            U_modes = np.square(np.linalg.norm(self._get("qt")))* params["lq"]
            U += U_modes

        self._add("U", U)
        return U

    def _get_gradient(self, mode, params, psim, init=False, verbose=False):
        molt = self._get("molt")

        if "HMC" == mode:
            t = time.time()
            if isinstance(self.target, src.molecule.Density):
                dU_biased_x = get_grad_RMSD3(coord=molt.coords, psim=psim, pexp=self.target.data,
                                             size=self.target.size,
                                             sampling_rate=self.target.sampling_rate,
                                             sigma=self.target.gaussian_sigma, threshold=self.target.threshold)
            else:
                dU_biased_x = get_grad_RMSD3_img(coord=molt.coords, psim=psim, pexp=self.target.data,
                                             size=self.target.size,
                                             sampling_rate=self.target.sampling_rate,
                                             sigma=self.target.gaussian_sigma, threshold=self.target.threshold)
            t_biased = time.time() -t

            t = time.time()
            dU_potential_x = get_autograd(molt)
            t_potential = time.time() -t

            dU_positions = 2 *  self._get("xt")

            Fx = -((params["lb"] * dU_biased_x) + (params["lp"] * dU_potential_x) + (params["lx"] * dU_positions))
            if verbose : self._print_step(["t_biased", "t_potential"], [t_biased, t_potential])
            if init:
                self.fit["Fx"] = Fx
            else:
                self.fit["Fxt"] = Fx

        elif "HMCNMA" == mode:
            t = time.time()
            if isinstance(self.target, src.molecule.Density):
                dU_biased_x, dU_biased_q = get_grad_RMSD3_NMA(coord=molt.coords, psim=psim,
                                                              pexp=self.target.data,
                                                              size=self.target.size, A_modes=self.init.modes,
                                                              sampling_rate=self.target.sampling_rate,
                                                              sigma=self.target.gaussian_sigma,
                                                              threshold=self.target.threshold)
            else:
                dU_biased_x, dU_biased_q = get_grad_RMSD3_NMA_img(coord=molt.coords, psim=psim, pexp=self.target.data,
                                             size=self.target.size, A_modes=self.init.modes,
                                             sampling_rate=self.target.sampling_rate,
                                             sigma=self.target.gaussian_sigma, threshold=self.target.threshold)
            t_biased = time.time() - t

            t = time.time()
            dU_potential_x, dU_potential_q = get_autograd_NMA(self.init,  self._get("xt"),  self._get("qt"), self.init.modes)
            t_potential = time.time() - t

            dU_modes = 2 *  self._get("qt")
            dU_positions = 2 *  self._get("xt")

            Fx = -((params["lb"] * dU_biased_x) + (params["lp"] * dU_potential_x) + (params["lx"] * dU_positions))
            Fq = -((params["lb"] * dU_biased_q) + (params["lp"] * dU_potential_q) + (params["lq"] * dU_modes))

            if verbose: self._print_step(["t_biased", "t_potential"], [t_biased, t_potential])
            if init:
                self.fit["Fx"] = Fx
                self.fit["Fq"] = Fq
            else:
                self.fit["Fxt"] = Fx
                self.fit["Fqt"] = Fq

        elif "NMA" == mode:
            if isinstance(self.target, src.molecule.Density):
                _, dU_biased_q = get_grad_RMSD3_NMA(coord=molt.coords, psim=psim, pexp=self.target.data,
                                                          size=self.target.size, A_modes=self.init.modes,
                                                          sampling_rate=self.target.sampling_rate,
                                                          sigma=self.target.gaussian_sigma,
                                                          threshold=self.target.threshold)
            else:
                _, dU_biased_q = get_grad_RMSD3_NMA_img(coord=molt.coords, psim=psim, pexp=self.target.data,
                                             size=self.target.size, A_modes=self.init.modes,
                                             sampling_rate=self.target.sampling_rate,
                                             sigma=self.target.gaussian_sigma, threshold=self.target.threshold)
            dU_modes = 2 *  self._get("qt")
            Fq = -((params["lb"] * dU_biased_q) + (params["lq"] * dU_modes))
            if init:
                self.fit["Fq"] = Fq
            else:
                self.fit["Fqt"] = Fq

        elif "ROT" == mode:
            _, dU_biased_angles = get_grad_RMSD2_img_rot(coord=molt.coords, psim=psim, pexp=self.target.data,
                                             size=self.target.size, sampling_rate=self.target.sampling_rate,
                                             sigma=self.target.gaussian_sigma, angles=self._get("anglest"),
                                                                   coord0 = self._get("coord0"))
            Fangles = -(params["lb"] * dU_biased_angles)
            if init:
                self.fit["Fangles"] = Fangles
            else:
                self.fit["Fanglest"] = Fangles

    def _get_kinetic(self, mode, params):
        K=0
        if "HMC" in mode:
            K += 1 / 2 * src.constants.CARBON_MASS*np.sum(np.square(self._get("vt")))
        if "NMA" in mode:
            K += 1 / 2 * np.sum(np.square(self._get("wt")))
        if "ROT" in mode:
            K += 1 / 2 * np.sum(np.square(self._get("vanglest")))
        self.fit["K"].append(K)
        return K

    def _get_criterion(self, mode, params):
        C = 0
        if params["criterion"]:
            if "HMC" in mode:
                C += np.dot((self._get("xt").flatten() - self._get("x").flatten()), self._get("vt").flatten())
            if "NMA" in mode:
                C += np.dot((self._get("qt") - self._get("q")), self._get("wt"))
            self.fit["C"].append(C)
        else:
            self.fit["C"].append(0)
        return C

    def _get_density(self):
        if isinstance(self.target, src.molecule.Density):
            return self._get("molt").to_density(size=self.target.size, sampling_rate=self.target.sampling_rate,
                                     gaussian_sigma=self.target.gaussian_sigma, threshold=self.target.threshold,
                                     box_size=False).data
        else:
            return self._get("molt").to_image(size=self.target.size, sampling_rate=self.target.sampling_rate,
                                     gaussian_sigma=self.target.gaussian_sigma, threshold=self.target.threshold).data

    def _print_step(self, keys, values):
        s = ""
        for i in range(len(keys)):
            s += keys[i] + "=" + str(values[i]) + " ; "
        print(s)

    def _initialize(self, mode, params):
        # Initial coordinates
        coordt = np.array(self.init.coords)
        if "HMC" in mode:
            self.fit["xt"].append(self._get("x"))
            self.fit["vt"].append(np.random.normal(0, params["m_vt"], self.init.coords.shape))
            coordt += self._get("xt")
        if "NMA" in mode:
            self.fit["qt"].append(self._get("q"))
            self.fit["wt"].append(np.random.normal(0, params["m_wt"], self.init.modes.shape[1]))
            coordt += np.dot(self._get("qt"), self.init.modes)
        if "ROT" in mode:
            self.fit["anglest"].append(self._get("angles"))
            self.fit["vanglest"].append(np.random.normal(0, params["m_anglest"], 3))
            self.fit["coord0"] = coordt
            coordt = np.dot(coordt, generate_euler_matrix(self._get("anglest")))

        self.fit["molt"].append(src.molecule.Molecule(coordt, chain_id=self.init.chain_id))

    def _update_positions(self, mode, params):
        coordt = np.array(self.init.coords)
        if "HMC" in mode:
            self.fit["xt"].append(self._update_pstep(self._get("xt"), self._get("vt") ,params["dxt"], self._get("Fx")))
            coordt += self._get("xt")
        if "NMA" in mode:
            self.fit["qt"].append(self._update_pstep(self._get("qt"), self._get("wt") ,params["dqt"], self._get("Fq")))
            coordt += np.dot(self._get("qt"), self.init.modes)
        if "ROT" in mode:
            self.fit["anglest"].append(self._update_pstep(self._get("anglest"), self._get("vanglest") ,params["danglest"], self._get("Fangles")))
            self.fit["coord0"] = coordt
            coordt = np.dot(coordt, generate_euler_matrix(self._get("anglest")))
        self.fit["molt"].append(src.molecule.Molecule(coordt, chain_id=self.init.chain_id))

    def _update_pstep(self, x, v, dx, F):
        return x+ dx*v + dx**2 *(F/2)

    def _update_velocities(self, mode, params):
        if "HMC" in mode:
            self.fit["vt"].append(self._get("vt") + (params["dxt"] * (self._get("Fx") + self._get("Fxt")) / 2))
            self.fit["Fx"] = self._get("Fxt")
        if "NMA" in mode:
            self.fit["wt"].append(self._get("wt") + (params["dqt"] * (self._get("Fq") + self._get("Fqt")) / 2))
            self.fit["Fq"] = self._get("Fqt")
        if "ROT" in mode:
            self.fit["vanglest"].append(self._get("vanglest") + (params["danglest"] * (self._get("Fangles") + self._get("Fanglest")) / 2))
            self.fit["Fangles"] = self._get("Fanglest")

    def _update_vstep(self, v, dx, F, Ft):
        return v + dx*((F+Ft)/2)

    def _acceptation(self, mode,H, H_init):
        accept_p = np.min([1, np.exp((H_init - H)/H_init)])
        print("H<H_init="+str(H_init > H) +" ; H" + str(H)+" ; H_init" + str(H_init))
        if accept_p > np.random.rand() :
            print("ACCEPTED " + str(accept_p))
            if "HMC" in mode:
                self.fit["x"].append(self._get("xt"))
            if "NMA" in mode:
                self.fit["q"].append(self._get("qt"))
            if "ROT" in mode:
                self.fit["angles"].append(self._get("anglest"))
            self.fit["mol"].append(self._get("molt"))
        else:
            print("REJECTED " + str(accept_p))
            if "HMC" in mode:
                self.fit["x"].append(self._get("x"))
            if "NMA" in mode:
                self.fit["q"].append(self._get("q"))
            if "ROT" in mode:
                self.fit["angles"].append(self._get("angles"))
            self.fit["mol"].append(self._get("mol"))


    def HMC_step(self, mode , params, verbose=False):

    # Initial coordinates
        self._initialize(mode, params)
    # initial density
        psim = self._get_density()
    # Initial Potential Energy
        U = self._get_energy(mode, params, psim)
    # Initial gradient
        self._get_gradient(mode, params, psim, init=True)
    # Initial Kinetic Energy
        K = self._get_kinetic(mode, params)
    # Initial Hamiltonian
        H_init = U + K
        criterion = 0
        self.fit["L"].append(0)

        while (criterion >= 0 and self._get("L")< params["max_iter"]):
            tt = time.time()
        # Coordinate update
            self._update_positions(mode, params)
        # Density update
            t = time.time()
            psim = self._get_density()
            t_density = time.time() - t
        # CC update
            self.fit["CC"].append(cross_correlation(psim, self.target.data))
        # Potential energy update
            t = time.time()
            U = self._get_energy(mode, params, psim)
            t_energy = time.time() - t
        # Gradient Update
            t = time.time()
            self._get_gradient(mode, params, psim, verbose=verbose)
            t_gradient = time.time() - t
        # velocities update
            self._update_velocities(mode, params)
        # Kinetic update
            K = self._get_kinetic(mode, params)
            T = 2 * K / (src.constants.K_BOLTZMANN * 3* self.init.n_atoms)
        # criterion update
            criterion = self._get_criterion(mode, params)
            self.fit["L"][-1] += 1
        # Prints
            t_total = time.time() -tt
            self._print_step(["ITER" , "U_biased", "U_potential", "K", "T",  "C", "CC", "Time"],
                       [self._get("L"), self._get("U_biased"),self._get("U_potential"),self._get("K"),T, self._get("C"),self._get("CC"), t_total])
            # self._print_step(["angles", "Fangles"], [self._get("angles"), self._get("Fangles")])
            if verbose:
                self._print_step(["t_density", "t_gradient", "t_energy", "t_process", "t_total", ],
                                 [t_density, t_gradient, t_energy, t_total-(t_density+ t_gradient+ t_energy), t_total])
    # Hamiltonian update
        H = U + K
        print("TEST")
        print(U)
        print(K)
    # Metropolis acceptation
        self._acceptation(mode,H, H_init)

    def show(self):
        fig, ax = plt.subplots(2, 3, figsize=(10, 5))
        ax[0, 0].plot(self.fit['U'])
        ax[0, 1].plot(self.fit['U_potential'])
        ax[0, 2].plot(self.fit['U_biased'])
        ax[1, 0].plot(np.array(self.fit['K']) + np.array(self.fit['U']))
        ax[1, 0].plot(self.fit['U'])
        ax[1, 0].plot(self.fit['K'])
        ax[1, 1].plot(self.fit['C'])
        ax[1, 2].plot(self.fit['CC'])

        ax[0, 0].set_title('U')
        ax[0, 1].set_title('U_potential')
        ax[0, 2].set_title('U_biased')
        ax[1, 0].set_title('H=U+K')
        ax[1, 1].set_title('C')
        ax[1, 2].set_title('CC')

        fig.tight_layout()

    def show_3D(self, genfile):
        src.viewers.chimera_fit_viewer(self.res["mol"], self.target, genfile=genfile)

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(file=f, obj=self)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            fit = pickle.load(file=f)
            return fit






