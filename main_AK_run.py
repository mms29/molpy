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
cutoff=6.0
targets = []
targets_dens = []
f = 4
N=200//f
for i in range(N):
    mol =Molecule("/home/guest/Workspace/Genesis/FlexibleFitting/output/1ake24ake_"+str(i)+".pdb")
    # mol =Molecule("data/1ake24ake/1ake24ake_"+str(int(i*4))+".pdb")
    mol.center()
    targets.append(mol)
    targets_dens.append(Volume.from_coords(coord=mol.coords, size=size, cutoff=cutoff, sigma=sigma, voxel_size=voxel_size))


params ={
    "initial_biasing_factor" : 50,
    "n_step": 10,

    "local_dt" : 1e-15,
    "temperature" : 300,

    "global_dt" : 0.05,
    "rotation_dt" : 0.0001,
    "shift_dt" : 0.001,
    "n_iter":200,
    "n_warmup":180,
    "potentials" : ["bonds", "angles", "dihedrals"],
}
verbose = 0
n_chain= 2
fits_x=[]
fits_q=[]
fits_a=[]

for i in range(N):
    params["target_coords"]=targets[i].coords
    fits_x.append(FlexibleFitting(init = init, target= targets_dens[i], vars=[FIT_VAR_LOCAL, FIT_VAR_ROTATION, FIT_VAR_SHIFT], params=params, n_chain=n_chain, verbose=verbose,prefix="results/1ake24ake/fit_x"+str(i)))
    fits_q.append(FlexibleFitting(init = init, target= targets_dens[i], vars=[FIT_VAR_GLOBAL, FIT_VAR_ROTATION, FIT_VAR_SHIFT], params=params, n_chain=n_chain, verbose=verbose,prefix="results/1ake24ake/fit_q"+str(i)))
    fits_a.append(FlexibleFitting(init = init, target= targets_dens[i], vars=[FIT_VAR_LOCAL,FIT_VAR_GLOBAL, FIT_VAR_ROTATION, FIT_VAR_SHIFT], params=params, n_chain=n_chain, verbose=verbose,prefix="results/1ake24ake/fit_a"+str(i)))

multiple_fitting(models = fits_x + fits_q + fits_a, n_proc = 150 ,n_chain=n_chain)

#
# data=[]
# for i in range(N):
#     data.append(targets[i].coords.flatten())
# for i in range(N):
#     data.append(Molecule("results/fit_x"+str(i)+"_output.pdb").coords.flatten())
# for i in range(N):
#     data.append(Molecule("results/fit_q"+str(i)+"_output.pdb").coords.flatten())
# for i in range(N):
#     data.append(Molecule("results/fit_a"+str(i)+"_output.pdb").coords.flatten())
#
# for i in range(100):
#     q = np.array([2,-1,0]) * (i -50) * 2
#     mol = nma_deform(init, q)
#     data.append(mol.coords.flatten())
# open = Molecule("data/AK/AK_open.pdb")
# data.append(open.coords.flatten())
# close = Molecule("data/AK/AK_close.pdb")
# data.append(close.coords.flatten())
#
# compute_pca(data=data, length=[N,N,N,N,100,1,1], n_components=3, labels=["Genesis Traj", "Local", "Global", "Local + Global", "NMA Traj", "Open", "Close"])
