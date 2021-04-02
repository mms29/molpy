import copy
from itertools import permutations
import numpy as np

import src.forcefield
import src.functions
import src.io
import src.viewers
from src.constants import *


class Molecule:
    """
    Atomic structure of a molecule
    """

    def __init__(self, pdb_file):
        """
        Contructor
        :param pdb_file: PDB file
        """
        data = src.io.read_pdb(pdb_file)

        self.coords = data["coords"]
        self.n_atoms = data["coords"].shape[0]
        self.atom = data["atom"]
        self.atomNum = data["atomNum"]
        self.atomName = data["atomName"]
        self.resName = data["resName"]
        self.chainName = data["chainName"]
        self.resNum = data["resNum"]
        self.elemName = data["elemName"]
        self.normalModeVec=None
        self.forcefield = None

    def copy(self):
        """
        Copy Molecule Object
        :return: Molecule
        """
        return copy.deepcopy(self)


    def get_energy(self, **kwargs):
        """
        Compute Potential energy of the object
        :return: the Total Potential energy
        """
        return src.forcefield.get_energy(coords=self.coords, forcefield=self.forcefield, **kwargs)

    def center(self):
        """
        Center the coordinates around 0,0,0
        """
        self.coords -= np.mean(self.coords, axis=0)

    def rotate(self, angles):
        """
        Rotate the coordinates
        :param angles: list of 3 Euler angles
        """
        R= src.functions.generate_euler_matrix(angles)
        self.coords = np.dot(R, self.coords.T).T
        if self.normalModeVec is not None:
            for i in range(self.n_atoms):
                self.normalModeVec[i] =  np.dot(R , self.normalModeVec[i].T).T

    def show(self):
        """
        Show the structure using matplotlib
        """
        src.viewers.molecule_viewer(self)

    def set_normalModeVec(self, files, **kwargs):
        """
        Set normal modes vectors to the object
        :param files: directory containing the normal modes
        :param n_modes: number of desired normal modes
        """
        if "selection" in kwargs:
            files = list(np.array(files)[np.array(kwargs["selection"])-1])
        self.normalModeVec = src.io.read_modes(files)
        if self.normalModeVec.shape[0] != self.n_atoms:
            raise RuntimeError("Modes vectors and coordinates do not match : ("+str(self.normalModeVec.shape[0])+") != ("+str(self.n_atoms)+")")

    def set_forcefield(self, **kwargs):
        """
        Set the force field structure and parameters for the Molecule.
        """
        self.forcefield = MoleculeForceField(mol=self,**kwargs)

    def save_pdb(self, file):
        """
        Save to PDB Format
        :param file: pdb file path
        """
        data = {
            "atom" : self.atom,
            "atomNum" : self.atomNum,
            "atomName" : self.atomName,
            "resName" : self.resName,
            "chainName" : self.chainName,
            "resNum" : self.resNum,
            "coords" : self.coords,
            "elemName" : self.elemName
        }
        src.io.save_pdb(data = data, file=file)

    def allatoms2carbonalpha(self):
        carbonalpha_idx = np.where(self.atomName == "CA")[0]
        self.coords = self.coords[carbonalpha_idx]
        self.n_atoms = self.coords.shape[0]

        # Normal Modes
        if self.normalModeVec is not None:
            self.normalModeVec = self.normalModeVec[carbonalpha_idx]

        #PDB
        self.atom = self.atom[carbonalpha_idx]
        self.atomNum = self.atomNum[carbonalpha_idx]
        self.atomName = self.atomName[carbonalpha_idx]
        self.resName = self.resName[carbonalpha_idx]
        self.chainName = self.chainName[carbonalpha_idx]
        self.resNum = self.resNum[carbonalpha_idx]
        self.elemName = self.elemName[carbonalpha_idx]

    def allatoms2backbone(self):
        backbone_idx = []
        for i in range(len(self.atomName)):
            if not self.atomName[i].startswith("H"):
                backbone_idx.append(i)
        backbone_idx = np.array(backbone_idx)

        self.coords = self.coords[backbone_idx]
        self.n_atoms = self.coords.shape[0]

        # Normal Modes
        if self.normalModeVec is not None:
            self.normalModeVec = self.normalModeVec[backbone_idx]

        # Forcefield
        self.forcefield.mass = self.forcefield.mass[backbone_idx]
        self.forcefield.charge = self.forcefield.charge[backbone_idx]

        # Bonds
        new_bonds, bonds_idx = src.functions.select_idx(param=self.forcefield.bonds, idx=backbone_idx)
        self.forcefield.bonds = new_bonds
        self.forcefield.Kb = self.forcefield.Kb[bonds_idx]
        self.forcefield.b0 = self.forcefield.b0[bonds_idx]

        # Angles
        new_angles, angles_idx = src.functions.select_idx(param=self.forcefield.angles, idx=backbone_idx)
        self.forcefield.angles = new_angles
        self.forcefield.KTheta = self.forcefield.KTheta[angles_idx]
        self.forcefield.Theta0 = self.forcefield.Theta0[angles_idx]

        # Dihedrals
        new_dihedrals, dihedrals_idx = src.functions.select_idx(param=self.forcefield.dihedrals, idx=backbone_idx)
        self.forcefield.dihedrals = new_dihedrals
        self.forcefield.Kchi = self.forcefield.Kchi[dihedrals_idx]
        self.forcefield.delta = self.forcefield.delta[dihedrals_idx]
        self.forcefield.n = self.forcefield.n[dihedrals_idx]

        # Impropers
        new_impropers, impropers_idx = src.functions.select_idx(param=self.forcefield.impropers, idx=backbone_idx)
        self.forcefield.impropers = new_impropers
        self.forcefield.Kpsi = self.forcefield.Kchi[impropers_idx]
        self.forcefield.psi0 = self.forcefield.delta[impropers_idx]

        #PDB
        self.atom = self.atom[backbone_idx]
        self.atomNum = self.atomNum[backbone_idx]
        self.atomName = self.atomName[backbone_idx]
        self.resName = self.resName[backbone_idx]
        self.chainName = self.chainName[backbone_idx]
        self.resNum = self.resNum[backbone_idx]
        self.elemName = self.elemName[backbone_idx]

class MoleculeForceField:
    """
    ForceField & Structure of the Molecule
    """

    def __init__(self, mol, **kwargs):
        if ("psf_file" in kwargs) and ("prm_file" in kwargs) :
            self.set_forcefield_psf(mol, kwargs["psf_file"], kwargs["prm_file"])
        else:
            self.set_forcefield_default(mol)


    def set_forcefield_psf(self, mol, psf_file, prm_file):

        psf = src.io.read_psf(psf_file)
        prm = src.io.read_prm(prm_file)

        self.bonds=psf["bonds"]
        self.angles=psf["angles"]
        self.dihedrals=psf["dihedrals"]
        self.impropers=psf["impropers"]
        atom_type = np.array(psf["atomNameRes"])

        self.n_bonds = len(self.bonds)
        self.n_angles = len(self.angles)
        self.n_dihedrals = len(self.dihedrals)
        self.n_impropers = len(self.impropers)

        self.Kb = np.zeros(self.n_bonds)
        self.b0 = np.zeros(self.n_bonds)
        self.KTheta = np.zeros(self.n_angles)
        self.Theta0 = np.zeros(self.n_angles)
        self.Kchi = np.zeros(self.n_dihedrals)
        self.n = np.zeros(self.n_dihedrals)
        self.delta = np.zeros(self.n_dihedrals)
        self.Kpsi = np.zeros(self.n_impropers)
        self.psi0 = np.zeros(self.n_impropers)
        self.charge = np.array(psf["atomCharge"])
        self.mass = np.array(psf["atomMass"])
        self.epsilon = np.zeros(mol.n_atoms)
        self.Rmin = np.zeros(mol.n_atoms)

        for i in range(self.n_bonds):
            comb = atom_type[self.bonds[i]]
            found = False
            for perm in [comb, comb[::-1]]:
                bond = "-".join(perm)
                if bond in prm["bonds"]:
                    self.Kb[i] = prm["bonds"][bond][0]
                    self.b0[i] = prm["bonds"][bond][1]
                    found = True
                    break
            if not found:
                raise RuntimeError("Enable to locale BONDS item in the PRM file")

        for i in range(self.n_angles):
            comb = atom_type[self.angles[i]]
            found = False
            for perm in [comb, comb[::-1]]:
                angle = "-".join(perm)
                if angle in prm["angles"]:
                    self.KTheta[i] = prm["angles"][angle][0]
                    self.Theta0[i] = prm["angles"][angle][1]
                    found = True
                    break
            if not found:
                raise RuntimeError("Enable to locale ANGLES item in the PRM file")

        for i in range(self.n_dihedrals):
            comb = list(atom_type[self.dihedrals[i]])
            found = False
            for perm in permutations(comb):
                dihedral = "-".join(perm)
                if dihedral in prm["dihedrals"]:
                    self.Kchi[i] = prm["dihedrals"][dihedral][0]
                    self.n[i] = prm["dihedrals"][dihedral][1]
                    self.delta[i] = prm["dihedrals"][dihedral][2]
                    found = True
                    break
            if not found:
                found = False
                for perm in permutations(comb):
                    dihedral = "-".join(['X'] + list(perm[1:3]) + ['X'])
                    if dihedral in prm["dihedrals"]:
                        self.Kchi[i] = prm["dihedrals"][dihedral][0]
                        self.n[i] = prm["dihedrals"][dihedral][1]
                        self.delta[i] = prm["dihedrals"][dihedral][2]
                        found = True
                        break
                if not found:
                    raise RuntimeError("Enable to locale DIHEDRAL item in the PRM file")

        for i in range(self.n_impropers):
            comb = list(atom_type[self.impropers[i]])
            found = False
            for perm in permutations(comb):
                improper = "-".join(perm)
                if improper in prm["impropers"]:
                    self.Kpsi[i] = prm["impropers"][improper][0]
                    self.psi0[i] = prm["impropers"][improper][1]
                    found = True
                    break
            if not found:
                found = False
                for perm in permutations(comb):
                    improper = "-".join([perm[0], 'X', 'X', perm[3]])
                    if improper in prm["impropers"]:
                        self.Kpsi[i] = prm["impropers"][improper][0]
                        self.psi0[i] = prm["impropers"][improper][1]
                        found = True
                        break
                if not found:
                    raise RuntimeError("Enable to locale IMPROPER item in the PRM file")

        for i in range(mol.n_atoms):
            if atom_type[i] in prm["nonbonded"]:
                self.epsilon[i] = prm["nonbonded"][atom_type[i]][0]
                self.Rmin[i] = prm["nonbonded"][atom_type[i]][1]
            else:
                raise RuntimeError("Enable to locale NONBONDED item in the PRM file")

    def set_forcefield_default(self, mol):

        chainName = mol.chainName
        chainSet = set(chainName)

        bonds = [[], []]
        angles = [[], [], []]
        dihedrals = [[], [], [], []]

        for i in chainSet:
            idx = np.where(chainName == i)[0]
            bonds[0] += list(idx[:-1])
            bonds[1] += list(idx[1:])

            angles[0] += list(idx[:-2])
            angles[1] += list(idx[1:-1])
            angles[2] += list(idx[2:])

            dihedrals[0] += list(idx[:-3])
            dihedrals[1] += list(idx[1:-2])
            dihedrals[2] += list(idx[2:-1])
            dihedrals[3] += list(idx[3:])

        self.bonds = np.array(bonds).T
        self.angles = np.array(angles).T
        self.dihedrals = np.array(dihedrals).T
        self.Kb = np.ones(self.bonds.shape[0]) * K_BONDS
        self.b0 = np.ones(self.bonds.shape[0]) * R0_BONDS
        self.KTheta = np.ones(self.angles.shape[0]) * K_ANGLES
        self.Theta0 = np.ones(self.angles.shape[0]) * THETA0_ANGLES
        self.Kchi = np.ones(self.dihedrals.shape[0]) * K_TORSIONS
        self.n = np.ones(self.dihedrals.shape[0]) * N_TORSIONS
        self.delta = np.ones(self.dihedrals.shape[0]) * DELTA_TORSIONS
        self.charge = np.zeros(len(chainName))
        self.mass = np.ones(len(chainName)) * CARBON_MASS
