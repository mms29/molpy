from src.molecule import Molecule
from src.simulation import nma_deform
from src.flexible_fitting import *
from src.viewers import molecule_viewer, chimera_molecule_viewer
from src.density import Volume

########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
init =Molecule.from_file("data/AK/AK_PSF.pdb")
init.center_structure()
init.add_modes("data/AK/modes_psf/vec.", n_modes=4)


# init.set_forcefield(psf_file="data/AK/AK.psf")
init.select_atoms(pattern='CA')
init.set_forcefield()

q = [100,-100,0,0]
target = nma_deform(init, q)
target.rotate([0.17,-0.13,0.23])

molecule_viewer([target, init])

size=64
sampling_rate=2.2
threshold= 4
gaussian_sigma=2
target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold)
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold)
target_density.show()

chimera_molecule_viewer([target, init])

########################################################################################################
#               HMC
########################################################################################################

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

    "angles_dt": 0.00005,
    "angles_mass": 1,
    "angles_init": np.zeros(3),
    "langles": 100,

    "shift_dt": 0.00005,
    "shift_mass": 1,
    "shift_init": np.zeros(3),
    "lshift" : 100,

}
n_iter=20
n_warmup = n_iter // 2


fit  =FlexibleFitting(init = init, target= target_density, vars=["x", "angles"], params=params,
                      n_iter=n_iter, n_warmup=n_warmup, n_chain=4, verbose=2)

fit.HMC()
fit.show()
fit.show_3D()

chimera_molecule_viewer([fit.res["mol"], target])

print(fit.res["angles"])