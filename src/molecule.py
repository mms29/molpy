import src.functions
import src.force_field
import numpy as np

class Atom:
    def __init__(self, coord, type):
        self.coord = coord
        self.type=type

class Molecule:

    def __init__(self, atoms):
        self.atoms = atoms
        self.n_atoms= len(atoms)

        self.coords = np.array([i.coord for i in atoms])
        self.bonds, self.angles, self.torsions = src.functions.cartesian_to_internal(self.coords)

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


    def select_atoms(self, pattern ='CA'):
        selected_atoms = []
        for i in range(self.n_atoms):
            if self.atoms[i].type ==  pattern:
                selected_atoms.append(self.atoms[i])
        return Molecule(selected_atoms)

    def show(self):
        src.functions.plot_structure(self.coords)


