import mkl
mkl.set_num_threads(1)
from src.molecule import Molecule
from src.simulation import nma_deform
from src.flexible_fitting import *
from src.viewers import molecule_viewer, chimera_molecule_viewer
from src.density import Volume
from src.constants import *
from src.functions import compute_pca

########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
init =Molecule("data/AK/AK_PSF.pdb")
init.center()
fnModes = np.array(["data/AK/modes_psf/vec."+str(i+7) for i in range(3)])
init.set_normalModeVec(fnModes)

init.set_forcefield(psf_file="data/AK/AK.psf", prm_file= "data/toppar/par_all36_prot.prm")

size=64
voxel_size=1.5
sigma=2
threshold=4
targets = []
targets_dens = []
N=80
for i in range(N):
    mol =Molecule("data/AK_run/AKrun_"+str(i)+".pdb")
    targets.append(mol)
    targets_dens.append(Volume.from_coords(coord=mol.coords, size=size, threshold=threshold, sigma=sigma, voxel_size=voxel_size))


params ={
    "initial_biasing_factor" : 50,
    "n_step": 20,

    "local_dt" : 2e-15,
    "temperature" : 300,

    "global_dt" : 0.05,
    "rotation_dt" : 0.0001,
    "shift_dt" : 0.001,
    "n_iter":20,
    "n_warmup":15,
    "potentials" : ["bonds", "angles", "dihedrals"],
}
verbose = 0
n_chain= 4
fits_x=[]
fits_q=[]
fits_a=[]
for i in range(N):
    print(i)
    params["target_coords"]=targets[i].coords
    fits_x.append(FlexibleFitting(init = init, target= targets_dens[i], vars=[FIT_VAR_LOCAL], params=params, n_chain=n_chain, verbose=verbose,prefix="results/AK_run/fit_x"))
    fits_q.append(FlexibleFitting(init = init, target= targets_dens[i], vars=[FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,prefix="results/AK_run/fit_q"))
    fits_a.append(FlexibleFitting(init = init, target= targets_dens[i], vars=[FIT_VAR_LOCAL,FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,prefix="results/AK_run/fit_q"))

multiple_fitting(models = fits_x + fits_q + fits_a, n_proc = 40 ,n_chain=n_chain)

#
# data=[]
# for i in range(N):
#     data.append(targets[i].coords.flatten())
# for i in range(100):
#     q = np.array([2,-1,0]) * (i -50) * 2
#     mol = nma_deform(init, q)
#     data.append(mol.coords.flatten())
# open = Molecule("data/AK/AK_open.pdb")
# data.append(open.coords.flatten())
# close = Molecule("data/AK/AK_close.pdb")
# data.append(close.coords.flatten())
#
# compute_pca(data=data, length=[N,100,1,1], n_components=2, labels=["Genesis Traj", "NMA Traj", "Open", "Close"])
