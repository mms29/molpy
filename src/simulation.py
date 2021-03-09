import src.functions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.molecule import Molecule
from src.forcefield import get_energy


class Simulator:

    def __init__(self, mol):
        self.deformed_mol = [mol]
        self.n_atoms = mol.n_atoms

    def nma_deform(self, q):
        if self.deformed_mol[0].modes is not None:
            n_modes = self.deformed_mol[0].modes.shape[1]
            self.q =q
            deformed_mol = Molecule.from_molecule(self.deformed_mol[0])
            deformed_mol.coords = np.dot(self.q, self.deformed_mol[-1].modes) + self.deformed_mol[-1].coords
            self.deformed_mol.append(deformed_mol)

            return self.deformed_mol[-1]

    def mc_deform(self, v_bonds=0.1, v_angles=0.01, v_torsions=0.01):
        internals, first = self.deformed_mol[-1].get_internal()

        if v_bonds != 0:
            internals[:,:,0] += np.random.normal(0,v_bonds, internals[:,:,0].shape)
        if v_angles != 0:
            internals[:,:,1] += np.random.normal(0,v_angles, internals[:,:,1].shape)
        if v_torsions != 0:
            internals[:,:,2] += np.random.normal(0,v_torsions, internals[:,:,2].shape)

        mol = Molecule.from_internals(internals, first=first, modes=self.deformed_mol[-1].modes, chain_id=self.deformed_mol[-1].chain_id)
        self.deformed_mol.append(mol)
        return mol

    def energy_min(self, U_lim, step, internal = False):
        deformed_mol = self.deformed_mol[-1]
        deformed_coords = deformed_mol.coords
        U_step = get_energy(deformed_mol, verbose=True)
        print("U_init = "+str(U_step))

        accept=[]
        while U_lim < U_step:
            candidate_coords = deformed_coords +  np.random.normal(0, step, (self.n_atoms,3))
            candidate_mol = Molecule(candidate_coords, chain_id=deformed_mol.chain_id)
            U_candidate = get_energy(candidate_mol, verbose=False)

            if U_candidate < U_step :
                U_step = U_candidate
                deformed_coords = candidate_coords
                accept.append(1)
                print("U_step = "+str(U_step)+ " ; acceptance_rate="+str(np.mean(accept)))
            else:
                accept.append(0)
                # print("rejected\r")
            if len(accept) > 20 : accept = accept[-20:]

        self.deformed_mol.append(Molecule(deformed_coords,chain_id=deformed_mol.chain_id, modes=self.deformed_mol[-1].modes))
        return self.deformed_mol[-1]



