import mkl
mkl.set_num_threads(1)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from src.molecule import Molecule
from src.simulation import nma_deform
from src.flexible_fitting import *
from src.viewers import molecule_viewer, chimera_molecule_viewer, chimera_fit_viewer
from src.density import Volume
from src.constants import *

########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
init =Molecule("data/ATPase/1su4_PSF_min.pdb")
init.center()
fnModes = np.array(["data/ATPase/1su4_modes_PSF/vec."+str(i+7) for i in range(5)])
init.set_normalModeVec(fnModes)
init.set_forcefield(psf_file="data/ATPase/1su4.psf",prm_file="data/toppar/par_all36_prot.prm")

target = Molecule("data/ATPase/1iwo_fitted_PSF.pdb")
target.rotate([-0.43815544 , 0.19210595 , 0.62026101])
target.center()

size=200
sampling_rate=2.0
cutoff= 6.0
gaussian_sigma=2
target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, cutoff=cutoff)
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, cutoff=cutoff)

params ={
    "biasing_factor" : 0.05,
    "potential_factor" : 1,
    "potentials":["bonds", "angles", "dihedrals", "impropers", "urey", "vdw","elec"],
    "cutoffpl": 10,
    "cutoffnb" : 7.0,

    "local_dt" : 2e-15,
    "temperature" : 300,
    "global_dt" : 0.1,
    "rotation_dt": 0.0001,
    "shift_dt": 0.001,
    # "shift_dt" : 0.0001,
    "n_iter":50,
    "n_warmup":45,
    "n_step": 100,
    "criterion": False,
    "target_coords" : target.coords,
    "limit" :100,
    "nb_update":20,
}
n_chain=4
verbose=2

fitx  =FlexibleFitting(init=init, target=target_density, vars=["local"], params=params, n_chain=n_chain, verbose=verbose,
                       prefix = "results/ATPase/fitx")
fita  =FlexibleFitting(init=init, target=target_density, vars=["local","global"], params=params, n_chain=n_chain, verbose=verbose,
                       prefix = "results/ATPase/fita")

fits = multiple_fitting(models=[fitx, fita], n_chain=n_chain, n_proc=10)
