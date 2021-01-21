import src.functions
import src.force_field
import numpy as np
import src.viewers


class Molecule:

    def __init__(self, coords, modes=None, atom_type=None):
        self.n_atoms= coords.shape[0]

        self.coords = coords
        self.bonds, self.angles, self.torsions = src.functions.cartesian_to_internal(self.coords)
        self.modes=modes
        self.atom_type = atom_type

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
            if self.atom_type[i] ==  pattern:
                selected_atoms.append(self.coords[i])
                if self.modes is not None:
                    selected_modes.append(self.modes[i])

        if self.modes is not None:
            selected_modes = np.array(selected_modes)
        selected_atoms =np.array(selected_atoms)

        return Molecule(selected_atoms , selected_modes)

    def center_structure(self):
        self.coords -= np.mean(self.coords)

    def show(self):
        src.viewers.structures_viewer(self)

    def to_density(self,size, gaussian_sigma, sampling_rate=1, threshold = None):
        halfN = int(size / 2)
        if ((self.coords < -size * sampling_rate / 2).any() or (self.coords > size * sampling_rate / 2).any()):
            print("WARNING !! box size = -" + str(np.max([
                (size * sampling_rate / 2) - np.max(self.coords),
                (size * sampling_rate / 2) + np.min(self.coords)]
            )))
        else:
            print("box size = " + str(np.max([
                (size * sampling_rate / 2) - np.max(self.coords),
                (size * sampling_rate / 2) + np.min(self.coords)]
            )))

        if threshold is not None :
            density = src.functions.volume_from_pdb_fast3(self.coords,size=size, sampling_rate=sampling_rate,
                                                          sigma=gaussian_sigma, threshold=threshold)
        else :
            density = src.functions.volume_from_pdb_fast2(self.coords, size=size, sampling_rate=sampling_rate,
                                                          sigma=gaussian_sigma)
        return Density(density, sampling_rate, gaussian_sigma)

    def to_image(self, size, gaussian_sigma, sampling_rate=1):
        image = np.zeros((size,size))
        for i in range(size):
            for j in range(size):
                mu = ((np.array([i, j]) - np.ones(2) * (size / 2)) * sampling_rate)
                image[i, j] = np.sum(
                        np.exp(-np.square(np.linalg.norm(self.coords[:,:2] - mu, axis=1)) / (2 * (gaussian_sigma ** 2))))
        return Image(image, sampling_rate, gaussian_sigma)

    def rotate(self, angles):
        a, b, c = angles
        cos = np.cos
        sin = np.sin
        R = [[cos(a) * cos(b), cos(a) * sin(b) * sin(c) - sin(a) * cos(c), cos(a) * sin(b) * cos(c) + sin(a) * sin(c)],
             [sin(a) * cos(b), sin(a) * sin(b) * sin(c) + cos(a) * cos(c), sin(a) * sin(b) * cos(c) - cos(a) * sin(c)],
             [-sin(b), cos(b) * sin(c), cos(b) * cos(c)]];
        for i in range(self.n_atoms):
            self.coords[i] = np.dot(self.coords[i], R)
            if self.modes is not None :
                self.modes[i] =  np.dot(self.modes[i], R)

    def show_internal(self):
        src.viewers.internal_viewer(self)

class Density:

    def __init__(self, data, sampling_rate, gaussian_sigma):
        self.data =data
        self.sampling_rate = sampling_rate
        self.gaussian_sigma = gaussian_sigma
        self.size = data.shape[0]

    def show(self):
        src.viewers.density_viewer(self)

class Image:

    def __init__(self, data, sampling_rate, gaussian_sigma):
        self.data =data
        self.sampling_rate = sampling_rate
        self.gaussian_sigma = gaussian_sigma
        self.size = data.shape[0]

    def show(self):
        src.viewers.image_viewer(self)

