# force numpy to use 1 thread per operation (It speeds up the computation)
import mkl
mkl.set_num_threads(1)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from src.molecule import Molecule
from src.density import Volume
import numpy as np
from src.flexible_fitting import FlexibleFitting, multiple_fitting
from src.viewers import chimera_fit_viewer, chimera_molecule_viewer

init = Molecule("data/P97/5ftm_psf.pdb")
init.center()
fnModes = np.array(["data/P97/modes_5ftm_psf/vec."+str(i+7) for i in range(4)])
init.set_normalModeVec(fnModes)
init.set_forcefield(psf_file="data/P97/5ftm.psf", prm_file="data/toppar/par_all36_prot.prm")
# init.get_energy(verbose=True)
# chimera_molecule_viewer([init])

size=128
voxel_size=1.4576250
cutoff=6.0
gaussian_sigma=2
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, cutoff=cutoff)

target= Molecule("data/P97/5ftn_PSF.pdb")
target.center()
# target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma,
#                                     cutoff=cutoff)
target_density = Volume.from_file('data/P97/emd_3299_128_filtered.mrc',voxel_size=voxel_size, sigma=gaussian_sigma, cutoff=cutoff)
target_density.data = (target_density.data / target_density.data.max())* init_density.data.max()
target_density.resize(200)

# target_density.rescale(init_density, "opt")
# target_density.compare_hist(init_density)

# from src.simulation import nma_deform
# target = nma_deform(init, [0,0,0,-1500])
# target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, cutoff=cutoff)
# chimera_molecule_viewer([init, target])
# chimera_fit_viewer(mol=init, target=target_density)

# target_density.resize(200)

params ={
    "initial_biasing_factor" : 0.05,
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
    "n_iter":5,
    "n_warmup":4,
    "n_step": 1000,
    "criterion": False,
    "target" : target,
    "limit" :100,
    "nb_update":20,
}
n_chain=4
verbose=2

fitx  =FlexibleFitting(init=init, target=target_density, vars=["local"], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/P97/fitx_exp")
fita  =FlexibleFitting(init=init, target=target_density, vars=["local", "global"], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/P97/fita_exp")
params["n_step"]=10
params["n_iter"]=500
params["n_warmup"]=450
params["potentials"]=["bonds", "angles", "dihedrals"]
fitq  =FlexibleFitting(init=init, target=target_density, vars=["global","shift"], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/P97/fitq_exp")
fits = multiple_fitting(models=[fitx, fita, fitq], n_chain=n_chain, n_proc=25)
