
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

gaussian_sigma = 3
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

            'torsion_sigma': 1,
            'k_torsions': src.constants.K_TORSIONS,
            'n_torsions': src.constants.N_TORSIONS,
            'delta_torsions': src.constants.DELTA_TORSIONS,
            'k_U': src.constants.CARBON_MASS/(src.constants.K_BOLTZMANN * 1000),
            'R_sigma' : 0.1,
            'shift_sigma' :1,
            'max_shift': 10,
            'verbose':1,
    #modes
            # 'n_modes': modes.shape[1],
            # 'A': modes,
            # 'q_sigma': 50,
            # 'q_mu': 0,
            # 'q_max': 200,
    # EM density
             'N':target_density.n_voxels,
             'halfN':int(target_density.n_voxels/2),
             'gaussian_sigma':gaussian_sigma,
             'sampling_rate': target_density.sampling_rate,
             'em_density': target_density.data,
             'epsilon_density': np.max(target_density.data) / 10

}


fit =src.fitting.Fitting(input_data, "md_torsions")
fit.optimizing(n_iter=5000)
# fit.sampling(n_chain=4, n_iter=100, n_warmup=800)
fit.plot_structure(save="results/sampling_structure_torsions_pas_nma.png")
# fit.plot_error_map(N=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate, slice=8)
# fit.plot_nma(sim.q)
# fit.plot_lp()


########################################################################################################
#               OUTPUTS
########################################################################################################
print("\nINIT ...")
init = src.molecule.Molecule.from_coords(atoms)
init.get_energy()
src.io.save_pdb(init, "results/nma_init.pdb", "data/AK/AK.pdb")


print("\nDEFORMED ...")
deformed = src.molecule.Molecule.from_coords(sim.deformed_structure)
deformed.get_energy()
src.io.save_density(sim.deformed_density, sampling_rate, "results/nma_deformed.mrc", origin=-np.ones(3)*sampling_rate*n_voxels/2)

print("\nFITTED ...")
opt = src.molecule.Molecule.from_coords(fit.opt_results['x'])
opt.get_energy()
src.io.save_pdb(opt, "results/nma_fitted.pdb", "data/AK/AK.pdb")
