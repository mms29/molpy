# force numpy to use 1 thread per operation (It speeds up the computation)
# import mkl
# mkl.set_num_threads(1)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from src.molecule import Molecule
from src.simulation import nma_deform
from src.flexible_fitting import *
from src.viewers import molecule_viewer, chimera_molecule_viewer
from src.density import Volume
from src.constants import *
from src.functions import show_rmsd_fit, get_RMSD_coords
########################################################################################################
#               IMPORT FILES
########################################################################################################

N=100

# import PDB
init =Molecule.from_file("data/AK/AK.pdb")
init.center_structure()
# fnModes = np.array(["data/AK/modes/vec."+str(i+7) for i in range(3)])
# init.add_modes(fnModes)

# init.set_forcefield(psf_file="data/AK/AK.psf")
init.select_atoms(pattern='CA')
init.set_forcefield()

# q = [100,-100,0,]
# target = nma_deform(init, q)
# target.rotate([0.17,-0.13,0.23])

size=64
sampling_rate=2.2
threshold= 4
gaussian_sigma=2
# angles = np.array([[1.5,2.0,-2.0]])
angles = np.random.uniform(-np.pi/2, np.pi/2, (N,3))
targets=[]
target_densities=[]
for i in range(N):
    target = copy.deepcopy(init)
    target.rotate(angles[i])
    targets.append(target)
    target_densities.append(Volume.from_coords(coord=target.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold))

params ={
    "biasing_factor" : 100,
    "n_step": 20,
    "criterion": True,
    "n_iter":20,
    "n_warmup":10,
    "rotation_dt" : 0.0001,
    "rotation_mass": 10000,}
n_chain = 4
verbose=2

models10000= []
models1000= []
models100= []
models10= []
models1= []
for i in target_densities:
    params = copy.deepcopy(params)
    params["rotation_mass"] = 1
    models1.append(FlexibleFitting(init=init, target=i, vars=["rotation"], params=params, n_chain=n_chain, verbose=verbose))
    params = copy.deepcopy(params)
    params["rotation_mass"] = 10
    models10.append(FlexibleFitting(init=init, target=i, vars=["rotation"], params=params, n_chain=n_chain, verbose=verbose))
    params = copy.deepcopy(params)
    params["rotation_mass"] = 100
    models100.append(FlexibleFitting(init=init, target=i, vars=["rotation"], params=params, n_chain=n_chain, verbose=verbose))
    params = copy.deepcopy(params)
    params["rotation_mass"] = 1000
    models1000.append(FlexibleFitting(init=init, target=i, vars=["rotation"], params=params, n_chain=n_chain, verbose=verbose))
    params = copy.deepcopy(params)
    params["rotation_mass"] = 10000
    models10000.append(FlexibleFitting(init=init, target=i, vars=["rotation"], params=params, n_chain=n_chain, verbose=verbose))

# fit = models[0].HMC()
# fit.show()
# show_rmsd_fit(mol=targets[0], fit=fit)
models = models1 +  models10 +  models100 +  models1000 +  models10000

fits = multiple_fitting(models, n_chain=n_chain, n_proc=96)

rmsd1 =     [get_RMSD_coords(targets[i].coords, fits[0*N:1*N][i].res["mol"].coords) for i in range(N)]
rmsd10 =    [get_RMSD_coords(targets[i].coords, fits[1*N:2*N][i].res["mol"].coords) for i in range(N)]
rmsd100 =   [get_RMSD_coords(targets[i].coords, fits[2*N:3*N][i].res["mol"].coords) for i in range(N)]
rmsd1000 =  [get_RMSD_coords(targets[i].coords, fits[3*N:4*N][i].res["mol"].coords) for i in range(N)]
rmsd10000 = [get_RMSD_coords(targets[i].coords, fits[4*N:5*N][i].res["mol"].coords) for i in range(N)]

import matplotlib.pyplot as plt
import numpy as np

print("Mean 1 : "+str(np.mean(rmsd1)))
print("STD 1 : "+str(np.std(rmsd1)))
print("Mean 10 : "+str(np.mean(rmsd10)))
print("STD 10 : "+str(np.std(rmsd10)))
print("Mean 100 : "+str(np.mean(rmsd100)))
print("STD 100 : "+str(np.std(rmsd100)))
print("Mean 1000 : "+str(np.mean(rmsd1000)))
print("STD 1000 : "+str(np.std(rmsd1000)))
print("Mean 10000 : "+str(np.mean(rmsd10000)))
print("STD 10000 : "+str(np.std(rmsd10000)))

fig, ax = plt.subplots(1,1,)
ax.plot(rmsd1, label="1")
ax.plot(rmsd10, label="10")
ax.plot(rmsd100, label="100")
ax.plot(rmsd1000, label="1000")
ax.plot(rmsd10000, label="10000")
fig.savefig("results/rotation_mass.png")
