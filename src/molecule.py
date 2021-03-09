import src.functions
import src.forcefield
import numpy as np
import src.viewers
from itertools import permutations
import src.io
from src.constants import *
import copy

class Molecule:

    def __init__(self, coords, modes=None, atom_type=None, chain_id=None, genfile=None, coarse_grained=False):
        self.n_atoms= coords.shape[0]

        self.coords = coords
        self.modes=modes
        self.atom_type = atom_type
        self.genfile=genfile
        self.coarse_grained=coarse_grained
        if chain_id is None:
            self.chain_id = [0,self.n_atoms]
            self.n_chain=1
        else:
            self.chain_id=chain_id
            self.n_chain = len(chain_id) -1

    @classmethod
    def from_internals(cls, internals, first, chain_id, modes=None, atom_type=None, ):
        n_chain = internals.shape[0]
        n_atoms = internals.shape[1] +3
        coords = np.zeros((n_atoms*n_chain,3))
        for i in range(n_chain):
            coords[chain_id[i]:chain_id[i+1]] = src.functions.internal_to_cartesian(internal=internals[0], first=first[0])

        return cls(coords=coords, modes=modes, atom_type=atom_type,chain_id=chain_id)

    @classmethod
    def from_molecule(cls, mol):
        return copy.deepcopy(mol)


    def get_energy(self, verbose=False):
        return src.forcefield.get_energy(self, self.coords, verbose=verbose)

    def get_gradient(self):
        return src.forcefield.get_autograd(self)

    def add_modes(self, file, n_modes):
        self.modes = src.functions.read_modes(file, n_modes=n_modes)
        if self.modes.shape[0] != self.n_atoms:
            raise RuntimeError("Modes vectors and coordinates do not match : ("+str(self.modes.shape[0])+") != ("+str(self.n_atoms)+")")


    def select_modes(self, selected_modes):
        self.modes = self.modes[:, selected_modes]

    def get_chain(self, id):
        return self.coords[self.chain_id[id]:self.chain_id[id+1]]

    def select_atoms(self, pattern="CA"):
        self.coarse_grained = True
        atom_idx = np.where(self.atom_type == pattern)[0]
        self.coords = self.coords[atom_idx]
        self.n_atoms = self.coords.shape[0]
        self.chain_id= [np.argmin(np.abs(atom_idx - self.chain_id[i])) for i in range(self.n_chain)] + [self.n_atoms]
        if self.modes is not None:
            self.modes = self.modes[atom_idx]

    def center_structure(self):
        self.coords -= np.mean(self.coords)

    def show(self):
        src.viewers.structures_viewer(self)

    def rotate(self, angles):
        R= src.functions.generate_euler_matrix(angles)
        self.coords = np.dot(R, self.coords.T).T
        for i in range(self.n_atoms):
            if self.modes is not None :
                self.modes[i] =  np.dot(R , self.modes[i].T).T

    def rotate2(self, angles):
        a, b, c = angles
        cos = np.cos
        sin = np.sin
        x = self.coords[:,0] * (cos(a) * cos(b))                            + self.coords[:,1] * (sin(a) * cos(b))                            + self.coords[:,2]* (-sin(b))
        y = self.coords[:,0] * (cos(a) * sin(b) * sin(c) - sin(a) * cos(c)) + self.coords[:,1] * (sin(a) * sin(b) * sin(c) + cos(a) * cos(c)) + self.coords[:,2]* (cos(b) * sin(c))
        z = self.coords[:,0] * (cos(a) * sin(b) * cos(c) + sin(a) * sin(c)) + self.coords[:,1] * (sin(a) * sin(b) * cos(c) - cos(a) * sin(c)) + self.coords[:,2]* (cos(b) * cos(c))
        self.coords[:, 0] = x
        self.coords[:, 1] = y
        self.coords[:, 2] = z
    # def show_internal(self):
    #     src.viewers.internal_viewer(self)

    def get_internal(self):
        internals =np.zeros((self.n_chain, self.n_atoms-3, 3))
        first = np.zeros((self.n_chain, 3, 3))
        for i in range(self.n_chain):
            internals[i]  = src.functions.cartesian_to_internal(self.get_chain(i))
            first[i] = self.get_chain(i)[:3]
        return internals, first

    def energy_min(self, U_lim, step, verbose=True):
        U_step = self.get_energy()
        print("U_init = "+str(U_step))
        deformed_coords=np.array(self.coords)

        accept=[]
        while U_lim < U_step:
            candidate_coords = deformed_coords +  np.random.normal(0, step, (self.n_atoms,3))
            U_candidate = src.forcefield.get_energy(self, candidate_coords, verbose=False)

            if U_candidate < U_step :
                U_step = U_candidate
                deformed_coords = candidate_coords
                accept.append(1)
                if verbose : print("U_step = "+str(U_step)+ " ; acceptance_rate="+str(np.mean(accept)))
            else:
                accept.append(0)
                # print("rejected\r")
            if len(accept) > 20 : accept = accept[-20:]

        new_mol =Molecule.from_molecule(self)
        new_mol.coords = deformed_coords
        return new_mol

    def set_forcefield(self, psf_file=None):

        if psf_file is not None:
            psf = src.io.read_psf(psf_file)
            self.bonds= psf["bonds"]
            self.angles= psf["angles"]
            self.dihedrals= psf["dihedrals"]
            self.atoms= psf["atoms"]
            self.prm = MoleculeForcefieldPrm.from_prm_file(self, prm_file=PARAMETER_FILE)

        else:
            bonds     = [[],[]]
            angles    = [[],[],[]]
            dihedrals = [[],[],[],[]]

            for i in range(self.n_chain) :
                idx = np.arange(self.chain_id[i], self.chain_id[i+1])
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
            self.prm = MoleculeForcefieldPrm.from_default(self)

class MoleculeForcefieldPrm:

    def __init__(self, Kb, b0, KTheta, Theta0, Kchi, n, delta, charge, mass):
        self.Kb = Kb
        self.b0 = b0
        self.KTheta = KTheta
        self.Theta0 = Theta0
        self.Kchi = Kchi
        self.n = n
        self.delta = delta
        self.charge = charge
        self.mass  = mass

    @classmethod
    def from_prm_file(cls, mol, prm_file):
        charmm_force_field = src.io.read_prm(prm_file)

        atom_type = []
        for i in range(mol.n_atoms):
            atom_type.append(mol.atoms[i][2])
        atom_type = np.array(atom_type)

        n_bonds = len(mol.bonds)
        n_angles = len(mol.angles)
        n_dihedrals = len(mol.dihedrals)

        Kb = np.zeros(n_bonds)
        b0 = np.zeros(n_bonds)
        KTheta= np.zeros(n_angles)
        Theta0= np.zeros(n_angles)
        Kchi= np.zeros(n_dihedrals)
        n= np.zeros(n_dihedrals)
        delta= np.zeros(n_dihedrals)
        charge= np.array(mol.atoms)[:, 3].astype(float)
        mass= np.array(mol.atoms)[:, 4].astype(float)

        for i in range(n_bonds):
            comb = atom_type[mol.bonds[i]]
            found = False
            for perm in [comb, comb[::-1]]:
                bond = "-".join(perm)
                if bond in charmm_force_field["bonds"]:
                    Kb[i] = charmm_force_field["bonds"][bond][0]
                    b0[i] = charmm_force_field["bonds"][bond][1]
                    found = True
                    break
            if not found:
                print("Err")

        for i in range(n_angles):
            comb = atom_type[mol.angles[i]]
            found = False
            for perm in [comb, comb[::-1]]:
                angle = "-".join(perm)
                if angle in charmm_force_field["angles"]:
                    KTheta[i] = charmm_force_field["angles"][angle][0]
                    Theta0[i] = charmm_force_field["angles"][angle][1]
                    found = True
                    break
            if not found:
                print("Err")

        for i in range(n_dihedrals):
            comb = list(atom_type[mol.dihedrals[i]])
            found = False
            for perm in permutations(comb):
                dihedral = "-".join(perm)
                if dihedral in charmm_force_field["dihedrals"]:
                    Kchi[i] = charmm_force_field["dihedrals"][dihedral][0]
                    n[i] = charmm_force_field["dihedrals"][dihedral][1]
                    delta[i] = charmm_force_field["dihedrals"][dihedral][2]
                    found = True
                    break
            if not found:
                found = False
                for perm in permutations(comb):
                    dihedral = "-".join(['X'] + list(perm[1:3]) + ['X'])
                    if dihedral in charmm_force_field["dihedrals"]:
                        Kchi[i] = charmm_force_field["dihedrals"][dihedral][0]
                        n[i] = charmm_force_field["dihedrals"][dihedral][1]
                        delta[i] = charmm_force_field["dihedrals"][dihedral][2]
                        found = True
                        break
                if not found:
                    print("Err")

        return cls(Kb, b0, KTheta, Theta0, Kchi, n, delta, charge, mass)

    @classmethod
    def from_default(cls, mol):
        Kb = np.ones(mol.bonds.shape[0]) * K_BONDS
        b0 = np.ones(mol.bonds.shape[0]) * R0_BONDS
        KTheta = np.ones(mol.angles.shape[0]) * K_ANGLES
        Theta0 = np.ones(mol.angles.shape[0]) * THETA0_ANGLES
        Kchi = np.ones(mol.dihedrals.shape[0]) * K_TORSIONS
        n = np.ones(mol.dihedrals.shape[0]) * N_TORSIONS
        delta = np.ones(mol.dihedrals.shape[0]) * DELTA_TORSIONS
        charge = np.zeros(mol.n_atoms)
        mass = np.ones(mol.n_atoms) * CARBON_MASS

        return cls(Kb, b0, KTheta, Theta0, Kchi, n, delta, charge, mass)
