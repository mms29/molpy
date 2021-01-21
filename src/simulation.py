import src.functions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.molecule import Molecule
from src.force_field import get_energy


class Simulator:

    def __init__(self, mol):
        self.deformed_mol = [mol]
        self.n_atoms = mol.n_atoms

    def nma_deform(self, amplitude=200):
        if self.deformed_mol[0].modes is not None:
            n_modes = self.deformed_mol[0].modes.shape[1]
            if isinstance(amplitude, list):
                self.q= amplitude
            else:
                self.q =np.random.uniform(-amplitude, amplitude, n_modes)

            deformed_coords = np.zeros((self.n_atoms,3))
            deformed_coords = np.dot(self.q, self.deformed_mol[-1].modes) + self.deformed_mol[-1].coords

            self.deformed_mol.append(Molecule(deformed_coords, modes=self.deformed_mol[-1].modes))

            return self.deformed_mol[-1]

    def mc_deform(self, v_bonds=0.1, v_angles=0.01, v_torsions=0.01):
        bonds = self.deformed_mol[-1].bonds
        angles = self.deformed_mol[-1].angles
        torsions = self.deformed_mol[-1].torsions

        if v_bonds != 0:
            bonds += np.random.normal(0,v_bonds, bonds.shape)
        if v_angles != 0:
            angles += np.random.normal(0,v_angles, angles.shape)
        if v_torsions != 0:
            torsions += np.random.normal(0,v_torsions, torsions.shape)

        deformed_coords = src.functions.internal_to_cartesian(np.array([bonds[2:], angles[1:], torsions]).T,
                                            first = self.deformed_mol[-1].coords[:3])
        self.deformed_mol.append(Molecule(deformed_coords, modes=self.deformed_mol[-1].modes))
        return self.deformed_mol[-1]

    def energy_min(self, U_lim, step, internal = False):
        deformed_mol = self.deformed_mol[-1]
        deformed_coords = deformed_mol.coords
        U_step = get_energy(deformed_mol, verbose=False)
        print("U_init = "+str(U_step))

        accept=[]
        while U_lim < U_step:
            candidate_coords = deformed_coords +  np.random.normal(0, step, (self.n_atoms,3))
            candidate_mol = Molecule(candidate_coords)
            U_candidate = get_energy(candidate_mol, verbose=False)

            if U_candidate < U_step :
                U_step = U_candidate
                deformed_coords = candidate_coords
                accept.append(1)
                print("U_step = "+str(U_step)+ " ; acceptance_rate="+str(np.mean(accept)))
            else:
                accept.append(0)
            if len(accept) > 20 : accept = accept[-20:]

        self.deformed_mol.append(Molecule(deformed_coords, modes=self.deformed_mol[-1].modes))
        return self.deformed_mol[-1]



