from src.functions import *
import src.simulation
import src.fitting
import src.constants
import src.molecule
import src.io
from src.viewers import structures_viewer
import matplotlib.pyplot as plt

# import PDB
init =src.io.read_pdb("data/AK/AK.pdb")
init.add_modes("data/AK/modes/vec.", n_modes=3)
init.center_structure()
init = init.select_atoms(pattern='CA')
init.rotate(angles = [0,np.pi/2, np.pi/2])

sim = src.simulation.Simulator(init)
nma = sim.nma_deform( amplitude=[200,-100,0])
# target=nma
target = sim.mc_deform(v_bonds=0.01, v_angles=0.001, v_torsions = 0.01)

# target.show_internal()
structures_viewer([target, init, nma],names=["target", "init", "nma"])

#
# # IMAGE

gaussian_sigma=2
sampling_rate=3
n_pixels=32
target_image = target.to_image(n_pixels, gaussian_sigma=gaussian_sigma, sampling_rate=sampling_rate)
target_image.show()
init_image = init.to_image(n_pixels, gaussian_sigma=gaussian_sigma, sampling_rate=sampling_rate)
init_image.show()

# DENSITY
# gaussian_sigma=2
# sampling_rate=3
# n_voxels=32
# target_density = target.to_density(n_voxels, gaussian_sigma=gaussian_sigma, sampling_rate=sampling_rate)
# target_density.show()


########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

param = {
    'k_U': 0.1,#src.constants.CARBON_MASS/(src.constants.K_BOLTZMANN * 1000),
    'torsion_sigma' : 0.1,
    'R_sigma' : 0.01,
    'shift_sigma' :0.1,
    'max_shift': 1,
    'q_sigma': 400,
    "verbose" : 0,
    "nu" : 10,
}


fit =src.fitting.Fitting(init, target_image, "MC_2D")
m = fit.optimizing(n_iter=1000, param=param)
fit.plot_structure(target=target)
m.to_image(n_pixels, gaussian_sigma=gaussian_sigma, sampling_rate=sampling_rate).show()
target_image.show()

fit.sampling(n_chain=4, n_iter=50, n_warmup=150, param=param)
# fit.plot_lp()


########################################################################################################
#               OUTPUTS
########################################################################################################
print("\nINIT ...")
init.get_energy()
src.io.save_pdb(init, "results/MCNMA_AK_init2.pdb", "data/AK/AK.pdb")


print("\nDEFORMED ...")
deformed = src.molecule.Molecule.from_coords(sim.deformed_structure)
deformed.get_energy()
deformed_density = deformed.to_density(n_voxels=n_voxels, sampling_rate=sampling_rate, sigma=gaussian_sigma)
src.io.save_density(deformed_density, "results/MCNMA_AK_target3.mrc")

print("\nFITTED ...")
opt = src.molecule.Molecule.from_coords(fit.opt_results['x'])
opt.get_energy()
opt.show()
src.io.save_pdb(opt, "results/MCNMA_AK_fitted2.pdb", "data/AK/AK.pdb")
