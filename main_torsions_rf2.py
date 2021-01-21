
from src.functions import *
import src.simulation
import src.fitting
import src.constants
import src.molecule
import src.io
import src.viewers
import matplotlib.pyplot as plt

init =src.io.read_pdb("data/ATPase/1iwo.pdb")
init.add_modes("data/ATPase/modes/vec.", n_modes=13)
init = init.select_atoms(pattern='CA')

target = src.io.read_pdb("data/ATPase/1su4_rotated.pdb").select_atoms(pattern='CA')
target.show()
src.viewers.structures_viewer([init, target])

gaussian_sigma = 2
target_density = target.to_density(n_voxels=32, sigma=gaussian_sigma, sampling_rate=8)
target_density.show()
src.io.save_density(target_density, "data/ATPase/deformed.mrc")

########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

input_data = {
    # structure
             'n_atoms': init.n_atoms,
             'y': target.coords,

             'epsilon': 1,

            'bonds': init.bonds[2:],
            'angles': init.angles[1:],
            'torsions': init.torsions,
            'first': init.coords[:3],

            'torsion_sigma': 0.1,
            'k_torsions': src.constants.K_TORSIONS,
            'n_torsions': src.constants.N_TORSIONS,
            'delta_torsions': src.constants.DELTA_TORSIONS,
            'k_U': src.constants.CARBON_MASS/(src.constants.K_BOLTZMANN * 1000),
            'R_sigma' : 0.1,
            'shift_sigma' :5,
            'max_shift': 50,
            'verbose':0,

             'gaussian_sigma':gaussian_sigma,
}


fit =src.fitting.Fitting(input_data, "md_torsions_emmap")
fit.optimizing(n_iter=10000)
# fit.sampling(n_chain=4, n_iter=100, n_warmup=800)
fit.plot_structure(save="results/sampling_structure_torsions_pas_nma.png")
# fit.plot_error_map(N=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate, slice=8)
# fit.plot_nma(sim.q)
# fit.plot_lp()


########################################################################################################
#               OUTPUTS
########################################################################################################
print("\nINIT ...")
init.get_energy()
src.io.save_pdb(init, "results/ATPASE_test_init.pdb", "data/ATPase/1iwo.pdb")


print("\nDEFORMED ...")
target.get_energy()
src.io.save_density(target_density, "results/ATPASE_test_target.mrc" )

print("\nFITTED ...")
opt = src.molecule.Molecule.from_coords(fit.opt_results['x'])
opt.get_energy()
src.io.save_pdb(opt, "results/ATPASE_test_fitted.pdb", "data/ATPase/1iwo.pdb")
