
############################################################
###  PRODY
############################################################
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from src.molecule import Molecule
import numpy as np
import copy
from src.viewers import chimera_molecule_viewer,chimera_fit_viewer
from src.constants import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.functions import compute_pca
from src.density import Volume
from src.flexible_fitting import FlexibleFitting, multiple_fitting

init = Molecule.from_file("data/AK/AK_prody/output.pdb")
init.center_structure()
fnModes = np.array(["data/AK/AK_prody/modes_psf/vec."+str(i+7) for i in range(6)])
init.add_modes(fnModes)
# init.select_atoms(pattern="CA")
init.set_forcefield("data/AK/AK_prody/output.psf", "../par_all36_prot.prm")
voxel_size=2.0
init_density = Volume.from_coords(coord=init.coords, voxel_size=voxel_size, size=64, threshold=5, sigma=2)

N=41
mols=[]
targets = []
for i in range(N):
    mol = Molecule.from_file("data/AK/AK_prody/frame_reduced_"+str(i)+".pdb")
    mols.append(mol)
    target = Volume.from_coords(coord=mol.coords, voxel_size=voxel_size, size=64, threshold=5, sigma=2)
    targets.append(target)

# target.compare_hist(init_density)
# chimera_fit_viewer(init, target)

params ={
    "biasing_factor" : 10,
    "potential_factor" : 1,

    "local_dt" : 0.01,
    "global_dt" : 0.1,
    "shift_dt" : 0.01,

    "n_iter":20,
    "n_warmup":10,
    "n_step": 20,
    "criterion": False,
}

n_chain = 4
n_proc = 48
verbose=0
models_global=[]
models_local =[]
models_glocal=[]
for i in targets:
    models_global.append(FlexibleFitting(init=init, target=i, vars=["global"], params=params,n_chain=n_chain, verbose=verbose))
    models_local .append(FlexibleFitting(init=init, target=i, vars=["local"], params=params,n_chain=n_chain, verbose=verbose))
    models_glocal.append(FlexibleFitting(init=init, target=i, vars=["local", "global"], params=params,n_chain=n_chain, verbose=verbose))

# n_test = 20
# models_global[n_test].HMC_chain()
# models_global[n_test].show()
# models_global[n_test].show_3D()
# chimera_molecule_viewer([init, models_global[n_test].res["mol"]])

fits_global = multiple_fitting(models_global, n_chain=n_chain, n_proc=n_proc)
fits_local = multiple_fitting(models_local, n_chain=n_chain, n_proc=n_proc)
fits_glocal = multiple_fitting(models_glocal, n_chain=n_chain, n_proc=n_proc)

# n=39
# chimera_fit_viewer(fits_glocal[n].res["mol"], targets[n])
# chimera_molecule_viewer([mols[0]]), mols[10], mols[20], mols[30], mols[40]])

pca_data = [i.coords.flatten() for i in mols] + [i.res["mol"].coords.flatten() for i in fits_global]+ \
                                                [i.res["mol"].coords.flatten() for i in fits_local]+ \
                                                [i.res["mol"].coords.flatten() for i in fits_glocal]
length= [N,N,N,N]
labels=["Ground Truth", "Global", "Local", "Global + Local"]
compute_pca(data=pca_data, length=length, labels= labels, n_components=3, save="results/AK_prody_pca.png")

