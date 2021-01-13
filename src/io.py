import numpy as np
import src.molecule

def read_pdb(file):
    atoms = []
    with open(file, "r") as file :
        end=False
        for line in file:
            l = line.split()
            if len(l) >0:
                if l[0] == 'ATOM':
                    if not end:
                        atoms.append(src.molecule.Atom(coord = [float(l[6]), float(l[7]), float(l[8])], type=l[2]))
                if l[0] == 'TER':
                    end = True

    return src.molecule.Molecule(atoms)