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
from src .io import create_psf

init = Molecule("data/RF2/1gqe_PSF.pdb")
# create_psf(pdb_file="data/RF2/1gqe_fitted.pdb", prefix="data/RF2/1gqe", topology_file="data/toppar/top_all36_prot.rtf")

fnModes = np.array(["data/RF2/modes/vec."+str(i+7) for i in range(4)])
init.set_normalModeVec(fnModes)
init.set_forcefield(psf_file="data/RF2/1gqe.psf", prm_file="data/toppar/par_all36_prot.prm")
# init.allatoms2backbone()
# init.get_energy(verbose=True, cutoff=2.0)
# chimera_molecule_viewer([init])


size=130
voxel_size=2.819999
cutoff=6.0
gaussian_sigma=2
target = Molecule("data/RF2/1mi6.pdb")
target.coords += ((130/2) - np.array([(20+80)/ 2, (40+100)/ 2, (20+80)/ 2])) * voxel_size
init.coords += ((130/2) - np.array([(20+80)/ 2, (40+100)/ 2, (20+80)/ 2])) * voxel_size
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, cutoff=cutoff)
target_density = Volume.from_file('data/RF2/emd_1010_fitted.mrc', sigma=2.0, cutoff=6.0)
target_density.data = (target_density.data / target_density.data.max())* init_density.data.max()

# target_density.rescale(init_density)
# target_density.compare_hist(init_density)
chimera_fit_viewer(mol=init, target=target_density)
# from src.simulation import nma_deform
# target = nma_deform(init, [0,0,0,-1500])
# target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, cutoff=cutoff)
# chimera_molecule_viewer([init, target])
# chimera_fit_viewer(mol=init, target=init_density)
# chimera_fit_viewer(mol=fita.fit[0].coord[-1], target=target_density)

# target_density.resize(200)

params ={
    # "initial_biasing_factor" : 0.5,
    "biasing_factor" : 0.05,
    "potential_factor" : 1,
    "potentials":["bonds", "angles", "dihedrals", "impropers", "urey", "vdw","elec"],
    "cutoffpl": 5.0,
    "cutoffnb" : 2.0,

    "local_dt" : 2e-15,
    "temperature" : 300,
    "global_dt" : 0.1,
    "rotation_dt": 0.0001,
    "shift_dt": 0.001,
    # "shift_dt" : 0.0001,
    "n_iter":50,
    "n_warmup":45,
    "n_step": 1000,
    "criterion": False,
    "target" : target,
    "limit" :100,
    "nb_update":20,
}
n_chain=4
verbose=2

fitx  =FlexibleFitting(init=init, target=target_density, vars=["local"], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/RF2/fitx")
fita  =FlexibleFitting(init=init, target=target_density, vars=["local", "global"], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/RF2/fita")
# fita.HMC_chain()
params["n_step"]=10
params["n_iter"]=5000
params["n_warmup"]=4500
params["potentials"]=["bonds", "angles", "dihedrals"]
fitq  =FlexibleFitting(init=init, target=target_density, vars=["global","shift", "rotation"], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/RF2/fitq")
fits = multiple_fitting(models=[fitx, fita, fitq], n_chain=n_chain, n_proc=25)
