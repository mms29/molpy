import numpy as np

import src.functions
from src.forcefield import get_energy
from src.molecule import Molecule


def nma_deform(mol, q):
    """
    deform molecule using NMA
    :param mol: Molecule to deform
    :param q: numpy array of M normal modes amplitudes
    :return: deormed Molecule
    """
    new_mol = Molecule.from_molecule(mol)
    new_mol.coords += np.dot(q, mol.modes)
    return new_mol

def energy_min(mol, U_lim, step, verbose=True):
    """
    Minimize the Potential Energy of the molecule by random walk
    :param U_lim: Energy limit to reach
    :param step: step of the random walk
    :param verbose: verbose level
    :return: the deformed molecule
    """
    U_step = mol.get_energy()
    print("U_init = "+str(U_step))
    deformed_coords=np.array(mol.coords)

    accept=[]
    while U_lim < U_step:
        candidate_coords = deformed_coords +  np.random.normal(0, step, (mol.n_atoms,3))
        U_candidate = src.forcefield.get_energy(mol, candidate_coords, verbose=False)

        if U_candidate < U_step :
            U_step = U_candidate
            deformed_coords = candidate_coords
            accept.append(1)
            if verbose : print("U_step = "+str(U_step)+ " ; acceptance_rate="+str(np.mean(accept)))
        else:
            accept.append(0)
            # print("rejected\r")
        if len(accept) > 20 : accept = accept[-20:]

    new_mol =Molecule.from_molecule(mol)
    new_mol.coords = deformed_coords
    return new_mol


