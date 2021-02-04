import src.functions
import src.force_field
import numpy as np
import src.viewers


class Molecule:

    def __init__(self, coords, modes=None, atom_type=None, chain_id=None):
        self.n_atoms= coords.shape[0]

        self.coords = coords
        self.modes=modes
        self.atom_type = atom_type
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


    def get_energy(self, verbose=False):
        return src.force_field.get_energy(self,verbose)

    def get_gradient(self):
        return src.force_field.get_autograd(self)

    def add_modes(self, file, n_modes):
        self.modes = src.functions.read_modes(file, n_modes=n_modes)

    def select_modes(self, selected_modes):
        self.modes = self.modes[:, selected_modes]

    def get_chain(self, id):
        return self.coords[self.chain_id[id]:self.chain_id[id+1]]

    def select_atoms(self, pattern ='CA'):
        selected_atoms = []
        if self.modes is not None:
            selected_modes = []
        else :
            selected_modes = None

        new_chain_id=[0]
        for j in range(self.n_chain):
            for i in range(self.chain_id[j],self.chain_id[j+1]):
                if self.atom_type[i] ==  pattern:
                    selected_atoms.append(self.coords[i])
                    if self.modes is not None:
                        selected_modes.append(self.modes[i])
            new_chain_id.append(len(selected_atoms))
        if self.modes is not None:
            selected_modes = np.array(selected_modes)
        selected_atoms =np.array(selected_atoms)

        return Molecule(coords=selected_atoms , modes=selected_modes, chain_id=new_chain_id)

    def center_structure(self):
        self.coords -= np.mean(self.coords)

    def show(self):
        src.viewers.structures_viewer(self)

    def to_density(self,size, gaussian_sigma, sampling_rate=1, threshold = None, box_size=True):
        if box_size :
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
        return Density(data=density, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold)

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

    # def show_internal(self):
    #     src.viewers.internal_viewer(self)

    def get_internal(self):
        internals =np.zeros((self.n_chain, self.n_atoms-3, 3))
        first = np.zeros((self.n_chain, 3, 3))
        for i in range(self.n_chain):
            internals[i]  = src.functions.cartesian_to_internal(self.get_chain(i))
            first[i] = self.get_chain(i)[:3]
        return internals, first

    def energy_min(self, U_lim, step):
        U_step = self.get_energy()
        print("U_init = "+str(U_step))
        deformed_coords=np.array(self.coords)

        accept=[]
        while U_lim < U_step:
            candidate_coords = deformed_coords +  np.random.normal(0, step, (self.n_atoms,3))
            candidate_mol = Molecule(candidate_coords, chain_id=self.chain_id)
            U_candidate = candidate_mol.get_energy()

            if U_candidate < U_step :
                U_step = U_candidate
                deformed_coords = candidate_coords
                accept.append(1)
                print("U_step = "+str(U_step)+ " ; acceptance_rate="+str(np.mean(accept)))
            else:
                accept.append(0)
                # print("rejected\r")
            if len(accept) > 20 : accept = accept[-20:]

        return Molecule(deformed_coords,chain_id=self.chain_id, modes=self.modes)

class Density:

    def __init__(self, data, sampling_rate, gaussian_sigma=None, threshold=None):
        self.data =data
        self.sampling_rate = sampling_rate
        self.gaussian_sigma = gaussian_sigma
        self.threshold=threshold
        self.size = data.shape[0]

    def show(self):
        src.viewers.density_viewer(self)

class Image:

    def __init__(self, data, sampling_rate, gaussian_sigma=None):
        self.data =data
        self.sampling_rate = sampling_rate
        self.gaussian_sigma = gaussian_sigma
        self.size = data.shape[0]

    def show(self):
        src.viewers.image_viewer(self)

