# force numpy to use 1 thread per operation (It speeds up the computation)
# import mkl
# mkl.set_num_threads(1)

import src.molecule
from src.density import Volume
import numpy as np
from src.flexible_fitting import FlexibleFitting, multiple_fitting
from src.viewers import chimera_fit_viewer, chimera_molecule_viewer

init =src.molecule.Molecule.from_file("data/P97/5ftm.pdb")
init.center_structure()
init.add_modes("data/P97/modes_atoms/vec.", n_modes=4)
init.select_atoms(pattern='CA')
init.set_forcefield()

size=128
voxel_size=1.458
threshold=4
gaussian_sigma=2
target_density = Volume.from_file('data/P97/emd_3299_128_filtered.mrc', voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
target_density.rescale(init_density)
target_density.compare_hist(init_density)

# from src.simulation import nma_deform
# target = nma_deform(init, [0,0,-1500,0])
# target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)

# chimera_fit_viewer(mol=init, target=target_density)


params ={
    "lb" : 200,
    "lp" : 1,

    "max_iter": 10,
    "criterion" :False,

    "x_dt" : 0.005,
    "x_mass" : 1,
    "x_init": np.zeros(init.coords.shape),

    "q_dt" : 0.05,
    "q_mass" : 1,
    "q_init": np.zeros(init.modes.shape[1]),

    "angles_dt": 0.00001,
    "angles_mass": 1,
    "angles_init": np.zeros(3),
    "langles": 100,

    "shift_dt": 0.00005,
    "shift_mass": 1,
    "shift_init": np.zeros(3),
    "lshift" : 100,

}
n_iter=10
n_chain=1
verbose=1

fitx  =FlexibleFitting(init=init, target=target_density, vars=["x", "shift"], params=params, n_iter=n_iter, n_warmup=n_iter // 2,n_chain=n_chain, verbose=verbose)
fitq  =FlexibleFitting(init=init, target=target_density, vars=["q", "shift"], params=params, n_iter=n_iter, n_warmup=n_iter // 2,n_chain=n_chain, verbose=verbose)
fitxq  =FlexibleFitting(init=init, target=target_density, vars=["x", "q", "shift"], params=params, n_iter=n_iter, n_warmup=n_iter // 2,n_chain=n_chain, verbose=verbose)
# fit.HMC_chain()
# fit.show_3D()
# chimera_molecule_viewer([fits[2].res["mol"], init])

fits = multiple_fitting(models=[fitx, fitq, fitxq], n_chain=n_chain, n_proc=8)


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
fig.tight_layout()



