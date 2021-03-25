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

init =src.molecule.Molecule.from_file("data/P97/5ftm_psf.pdb")
init.center_structure()
fnModes = np.array(["data/P97/modes_5ftm_psf/vec."+str(i+7) for i in range(4)])
init.add_modes(fnModes)
init.set_forcefield(psf_file="data/P97/5ftm.psf", prm_file="data/toppar/par_all36_prot.prm")

size=128
voxel_size=1.458
threshold=4
gaussian_sigma=2
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)

target_density = Volume.from_file('data/P97/emd_3299_128_filtered.mrc', voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
target_density.data = (target_density.data / target_density.data.max())* init_density.data.max()
# target_density.rescale(init_density, "opt")
target_density.compare_hist(init_density)

# from src.simulation import nma_deform
# target = nma_deform(init, [0,0,0,-1500])
# target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
# chimera_molecule_viewer([init, target])
# chimera_fit_viewer(mol=init, target=target_density)


params ={
    "initial_biasing_factor" : 100,
    "potential_factor" : 1,

    "local_dt" : 1e-15,
    "temperature" : 1000,
    "global_dt" : 0.05,
    # "shift_dt" : 0.0001,
    "n_iter":30,
    "n_warmup":20,
    "n_step": 20,
    "criterion": False,
}
n_chain=4
verbose=1
prefix = "results/p97_allatoms_exp100"
prefix_x =  prefix+"_fitx"
prefix_q =  prefix+"_fitq"
prefix_xq = prefix+"_fitxq"

fitx  =FlexibleFitting(init=init, target=target_density, vars=["local"], params=params, n_chain=n_chain, verbose=verbose, prefix=prefix_x)
fitq  =FlexibleFitting(init=init, target=target_density, vars=["global"], params=params, n_chain=n_chain, verbose=verbose, prefix=prefix_q)
fitxq  =FlexibleFitting(init=init, target=target_density, vars=["local", "global"], params=params, n_chain=n_chain, verbose=verbose,prefix=prefix_xq)

fits = multiple_fitting(models=[fitx, fitq, fitxq], n_chain=n_chain, n_proc=12)

fits[0].show(save=prefix_x + "_stats.png")
fits[1].show(save=prefix_q + "_stats.png")
fits[2].show(save=prefix_xq + "_stats.png")

import matplotlib.pyplot as plt
from src.functions import cross_correlation
from matplotlib.ticker import MaxNLocator
cc_init= cross_correlation(init_density.data, target_density.data)
L1 = np.cumsum(([1] + fits[0].fit[0]["L"])).astype(int) - 1
L2 = np.cumsum(([1] + fits[1].fit[0]["L"])).astype(int) - 1
L3 = np.cumsum(([1] + fits[2].fit[0]["L"])).astype(int) - 1
fig, ax = plt.subplots(1,1, figsize=(5,2))
ax.plot(np.array([cc_init]+fits[0].fit[0]["CC"]), '-', color="tab:red", label="x")
ax.plot(np.array([cc_init]+fits[1].fit[0]["CC"]), '-', color="tab:green", label="q")
ax.plot(np.array([cc_init]+fits[2].fit[0]["CC"]), '-', color="tab:blue", label="xq")
ax.set_ylabel("Correlation Coefficient")
ax.set_xlabel("HMC iteration")
ax.legend(loc="lower right", fontsize=9)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.savefig(prefix+ "_fits.png")



