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

init =src.molecule.Molecule.from_file("data/P97/5ftm.pdb")
# init.center_structure()
init.coords -= np.mean(init.coords)
fnModes = np.array(["data/P97/modes_atoms/vec."+str(i+7) for i in range(4)])
init.add_modes(fnModes)

init.select_atoms()
init.set_forcefield()
init.get_energy()

# init.set_forcefield(psf_file="data/P97/5ftm.psf", prm_file="data/toppar/par_all36_prot.prm")

size=128
voxel_size=1.458
threshold=4
gaussian_sigma=2
target_density = Volume.from_file('data/P97/emd_3299_128_filtered.mrc', voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)

target_density.data = (target_density.data / target_density.data.max())* init_density.data.max()
# target_density.rescale(init_density, "opt")
target_density.compare_hist(init_density)

# from src.simulation import nma_deform
# target = nma_deform(init, [0,0,0,-1500])
# target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
# chimera_molecule_viewer([init, target])
# chimera_fit_viewer(mol=init, target=target_density)


params ={
    "biasing_factor" : 200,
    "potential_factor" : 1,

    "local_dt" : 0.01,
    "temperature" : 3000,
    "local_sigma" : 0,
    "global_dt" : 0.05,
    # "shift_dt" : 0.0001,
    "n_iter":10,
    "n_warmup":5,
    "n_step": 10,
    "criterion": False,
}
n_chain=1
verbose=2

fitx  =FlexibleFitting(init=init, target=target_density, vars=["local"], params=params, n_chain=n_chain, verbose=verbose)
fitx.HMC_chain()
fitx.show()
fitx.show_3D()

fitq  =FlexibleFitting(init=init, target=target_density, vars=["global"], params=params, n_chain=n_chain, verbose=verbose)
fitxq  =FlexibleFitting(init=init, target=target_density, vars=["local", "global"], params=params, n_chain=n_chain, verbose=verbose)

fits = multiple_fitting(models=[fitx, fitq, fitxq], n_chain=n_chain, n_proc=24)

fits[0].show(save="results/p97_all_atoms_fitx2.png")
fits[1].show(save="results/p97_all_atoms_fitq2.png")
fits[2].show(save="results/p97_all_atoms_fitxq2.png")

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
fig.savefig("results/p97_all_atoms_fits2.png")



