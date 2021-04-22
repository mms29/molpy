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
from src.simulation import nma_deform

init = Molecule("data/P97/5ftm_psf.pdb")
init.center()
fnModes = np.array(["data/P97/modes_5ftm_psf/vec."+str(i+7) for i in range(4)])
init.set_normalModeVec(fnModes)
init.set_forcefield(psf_file="data/P97/5ftm.psf", prm_file="data/toppar/par_all36_prot.prm")
# init.get_energy(verbose=True)
# chimera_molecule_viewer([init])

size=128
voxel_size=2.0
cutoff=6.0
gaussian_sigma=2
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, cutoff=cutoff)

target=Molecule("data/P97/5ftm_synth_min.pdb")
target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma,
                                    cutoff=cutoff)


params ={
    "biasing_factor" : 0.1859,
    "potential_factor" : 1,
    "potentials":["bonds", "angles", "dihedrals"],
    "cutoffpl": 10,
    "cutoffnb" : 7.0,

    "local_dt" : 2e-15,
    "temperature" : 300,
    "global_dt" : 0.1,
    "rotation_dt": 0.0001,
    "shift_dt": 0.001,
    # "shift_dt" : 0.0001,
    "n_iter":20,
    "n_warmup":18,
    "n_step": 1000,
    "criterion": False,
    "target_coords" : target.coords
}
n_chain=4
verbose=2

fitx  =FlexibleFitting(init=init, target=target_density, vars=["local"], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/P97_synth/fitx")

fita  =FlexibleFitting(init=init, target=target_density, vars=["local", "global"], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/P97_synth/fita")

params["n_iter"] = 1000
params["n_step"] = 20
fitq  =FlexibleFitting(init=init, target=target_density, vars=["global", "rotation","shift"], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/P97_synth/fitq")
fits = multiple_fitting(models=[fitx, fita], n_chain=n_chain, n_proc=13)

#
# import matplotlib.pyplot as plt
# from src.functions import cross_correlation
# from matplotlib.ticker import MaxNLocator
# fig, ax = plt.subplots(1,1, figsize=(5,2))
# ax.plot(np.mean([i["CC"] for i in fits[0].fit], axis=0), '-', color="tab:red", label="local")
# ax.plot(np.mean([i["CC"] for i in fits[1].fit], axis=0), '-', color="tab:blue", label="global")
# ax.plot(np.mean([i["CC"] for i in fits[2].fit], axis=0), '-', color="tab:green", label="local + global")
# ax.set_ylabel("Correlation Coefficient")
# ax.set_xlabel("HMC iteration")
# ax.legend(loc="lower right", fontsize=9)
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# fig.savefig(prefix+ "_fits.png")



