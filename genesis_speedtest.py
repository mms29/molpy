import mkl
mkl.set_num_threads(1)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from src.molecule import Molecule
from src.density import Volume
from src.viewers import *
from src.flexible_fitting import FlexibleFitting

init = Molecule("data/1AKE/1ake_chainA_psf.pdb")
init.center()
init.set_normalModeVec(np.array(["data/1AKE/modes/vec." + str(i + 7) for i in range(3)]))
init.set_forcefield(psf_file="data/1AKE/1ake_chainA.psf", prm_file="data/toppar/par_all36_prot.prm")

target = Molecule("data/4AKE/4ake_fitted.pdb")
target.center()
target_density =Volume.from_coords(coord=target.coords, size=64, sigma=2.0, voxel_size=2.0, cutoff=6.0)

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
fit = FlexibleFitting(init=init, target=target_density, vars=["local", "global", "rotation", "shift"], params=params, n_chain=4,verbose=2, prefix="results/genesis_speedtest")
fit.HMC()
fit.show()