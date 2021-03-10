import time

import numpy as np

import src.io
from src.forcefield import *
from src.functions import *

init =src.io.read_pdb("data/AK_tomos/AK.pdb")
init.add_modes("data/AK_tomos/modes/vec.", n_modes=4)
init.select_atoms(pattern='CA')
init.set_forcefield()


x = np.zeros(3)
q = np.zeros(init.modes.shape[1])
angles = np.random.uniform(-0.5,0.5,3)


R = generate_euler_matrix(angles = angles)

t = time.time()
dangles1 =get_euler_autograd(coord = init.coords[0], A = init.modes[0], x=x, q=q, angles=angles)
print(time.time() - t)

t = time.time()
dangles2 =get_euler_grad(angles=angles, coord = init.coords[0])
print(time.time() - t)

print(dangles1[2] - np.sum(dangles2, axis=1))

d = {"x": np.array([0.0,0.0,0.0]), "angles": np.array([0.0,0.1,0.2])}
a = get_autograd(init, d)

########################################################################################################################

import src.simulation
from src.viewers import chimera_structure_viewer
from src.flexible_fitting import *
from src.molecule import Molecule
import copy


init =Molecule.from_file("data/AK/AK_PSF.pdb")
init.center_structure()
init.add_modes("data/AK/modes_psf/vec.", n_modes=4)
init.select_atoms(pattern='CA')
init.set_forcefield()

 # q = [300,-100,0,0]
q = [0,0,0,0]
# angles= [-0.1,0.2, 0.5]
target = src.simulation.nma_deform(init, q)
# target.rotate(angles)
shift = [8.3,5.2,7.0]
target.coords += shift

size=64
voxel_size=2.2
threshold= 4
gaussian_sigma=2
target_density = Image.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
init_density = Image.from_coords(coord=init.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
# init_density.show()
# target_density.show()

params ={
    "lb" : 100,
    "lp" : 1,
    "max_iter": 10,
    "criterion" :False,

    "q_init" : np.zeros(4),
    "q_dt" : 0.1,
    "q_mass" : 1,

    "angles_init" : np.zeros(3),
    "angles_dt" : 0.0001,
    "angles_mass" : 1,
    "langles" :0,

    "shift_init": np.zeros(3),
    "shift_dt": 0.001,
    "shift_mass": 1,
    "lshift": 0,
}
n_iter=20
n_warmup = n_iter // 2

fit  =FlexibleFitting(init = init, target= target_density, vars=["shift"], params=params,
                      n_iter=n_iter, n_warmup=n_warmup, n_chain=1, verbose=2)

fit.HMC()
fit.show()
fit.show_3D()
chimera_structure_viewer([fit.res["mol"], target, init])


test = copy.deepcopy(init)
test.rotate(fit.res["angles"])
chimera_structure_viewer([test, target])










import autograd.numpy as npg
from autograd import elementwise_grad

def get_energy_autograd(mol, params):
    coord = npg.array(mol.coords)
    if "x" in params:
        coord+= params["x"]
    if "q" in params:
        coord+= npg.dot(params["q"], mol.modes)
    if "angles" in params:
        coord = npg.dot(src.functions.generate_euler_matrix(params["angles"]), coord.T).T

    U_bonds = get_energy_bonds(coord, mol.bonds, mol.prm)
    U_angles = get_energy_angles(coord, mol.angles, mol.prm)
    U_dihedrals = get_energy_dihedrals(coord, mol.dihedrals, mol.prm)

    return U_bonds + U_angles + U_dihedrals

def get_autograd(mol, params):
    grad = elementwise_grad(get_energy_autograd, 1)
    return grad(mol, params)
d = {"x":np.array([0.0,0.0,0.0]),
     "q": np.array([0.0,0.0,0.0,0.0]),
     "angles" : np.array([0.0,0.0,0.0])}
get_energy_autograd(init,  d)
a = get_autograd(init,  d)

from src.forcefield import get_gradient_auto
get_gradient_auto(init, {"x": init.coords, "angles": [0.0,.0,.0]})