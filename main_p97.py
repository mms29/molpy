# force numpy to use 1 thread per operation (It speeds up the computation)
import mkl
mkl.set_num_threads(1)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import src.molecule
from src.density import Volume
import numpy as np
from src.flexible_fitting import FlexibleFitting, multiple_fitting
from src.viewers import chimera_fit_viewer, chimera_molecule_viewer

init =src.molecule.Molecule("data/P97/5ftm_psf.pdb")
init.center()
fnModes = np.array(["data/P97/modes_5ftm_psf/vec."+str(i+7) for i in range(4)])
init.set_normalModeVec(fnModes)
init.set_forcefield(psf_file="data/P97/5ftm.psf", prm_file="data/toppar/par_all36_prot.prm")
init.get_energy(verbose=True)
# chimera_molecule_viewer([init])

size=128
voxel_size=1.458
threshold=4
gaussian_sigma=2
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)

target_density = Volume.from_file('data/P97/emd_3299_128_filtered.mrc', voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
target_density.data = (target_density.data / target_density.data.max())* init_density.data.max()
# target_density.rescale(init_density, "opt")
# target_density.compare_hist(init_density)

from src.simulation import nma_deform
target = nma_deform(init, [0,0,0,-1500])
target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
# chimera_molecule_viewer([init, target])
# chimera_fit_viewer(mol=init, target=target_density)

target_density.resize(200)

params ={
    "initial_biasing_factor" : 10,
    "potential_factor" : 1,
    "potentials":["bonds", "angles", "dihedrals", "impropers", "vdw", "elec"],
    "cutoffpl": 25,
    "cutoffnb" : 20,

    "local_dt" : 2e-15,
    "temperature" : 300,
    "global_dt" : 0.05,
    # "shift_dt" : 0.0001,
    "n_iter":100,
    "n_warmup":80,
    "n_step": 40,
    "criterion": False,
}
n_chain=4
verbose=2
prefix = "results/p97_allatoms_synth"
prefix_x =  prefix+"_fitx"
prefix_q =  prefix+"_fitq"
prefix_xq = prefix+"_fitxq"

fitx  =FlexibleFitting(init=init, target=target_density, vars=["local"], params=params, n_chain=n_chain, verbose=verbose, prefix=prefix_x)
fitq  =FlexibleFitting(init=init, target=target_density, vars=["global"], params=params, n_chain=n_chain, verbose=verbose, prefix=prefix_q)
fitxq  =FlexibleFitting(init=init, target=target_density, vars=["local", "global"], params=params, n_chain=n_chain, verbose=verbose,prefix=prefix_xq)
fitxq.HMC()

# fits = multiple_fitting(models=[fitx, fitq, fitxq], n_chain=n_chain, n_proc=13)

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



