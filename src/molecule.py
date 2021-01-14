import src.functions
import src.force_field
import numpy as np
import src.viewers

class Atom:
    def __init__(self, coord, type):
        self.coord = coord
        self.type=type

class Molecule:

    def __init__(self, atoms, modes=None):
        self.atoms = atoms
        self.n_atoms= len(atoms)

        self.coords = np.array([i.coord for i in atoms])
        self.bonds, self.angles, self.torsions = src.functions.cartesian_to_internal(self.coords)
        self.modes=modes

    @classmethod
    def from_coords(cls, coords):
        atoms=[]
        for i in range(coords.shape[0]):
            atoms.append(Atom(coords[i], "No type"))
        return cls(atoms)

    def get_energy(self):
        return src.force_field.get_energy(self)

    def get_gradient(self):
        return src.force_field.get_gradient(self)

    def add_modes(self, file, n_modes):
        self.modes = src.functions.read_modes(file, n_modes=n_modes)

    def select_atoms(self, pattern ='CA'):
        selected_atoms = []
        if self.modes is not None:
            selected_modes = []
        else :
            selected_modes = None

        for i in range(self.n_atoms):
            if self.atoms[i].type ==  pattern:
                selected_atoms.append(self.atoms[i])
                if self.modes is not None:
                    selected_modes.append(self.modes[i])

        if self.modes is not None:
            selected_modes = np.array(selected_modes)

        return Molecule(selected_atoms , selected_modes)

    def center_structure(self):
        mean = np.mean(self.coords)
        for i in range(self.n_atoms):
            self.atoms[i].coord -= mean
        self.coords -= mean

    def show(self):
        src.viewers.structures_viewer(self)

    def to_density(self,n_voxels, sigma, sampling_rate=1):
        halfN = int(n_voxels / 2)
        if ((self.coords < -n_voxels * sampling_rate / 2).any() or (self.coords > n_voxels * sampling_rate / 2).any()):
            print("WARNING !! box size = -" + str(np.max([
                (n_voxels * sampling_rate / 2) - np.max(self.coords),
                (n_voxels * sampling_rate / 2) + np.min(self.coords)]
            )))
        else:
            print("box size = " + str(np.max([
                (n_voxels * sampling_rate / 2) - np.max(self.coords),
                (n_voxels * sampling_rate / 2) + np.min(self.coords)]
            )))
        n_atoms = self.coords.shape[0]
        em_density = np.zeros((n_voxels, n_voxels, n_voxels))
        for i in range(n_voxels):
            for j in range(n_voxels):
                for k in range(n_voxels):
                    mu = ((np.array([i, j, k]) - np.ones(3) * (n_voxels / 2)) * sampling_rate)
                    em_density[i, j, k] = np.sum(
                        np.exp(-np.square(np.linalg.norm(self.coords - mu, axis=1)) / (2 * (sigma ** 2))))
        return Density(em_density, sampling_rate)


class Density:

    def __init__(self, data, sampling_rate):
        self.data =data
        self.sampling_rate = sampling_rate
        self.n_voxels = data.shape[0]

    def show(self):
        src.viewers.density_viewer(self)


