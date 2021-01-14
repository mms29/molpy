from src.functions import *
import src.simulation
import src.fitting
import src.constants
import src.molecule
import src.io
import matplotlib.pyplot as plt

# import PDB
init =src.io.read_pdb("data/AK/AK.pdb")
init.add_modes("data/AK/modes/vec.", n_modes=3)
init = init.select_atoms(pattern='CA')

sim = src.simulation.Simulator(init.coords)


#GENERATE NMA DEFORMATIONS
nma_structure = sim.run_nma(modes = init.modes, amplitude=[200,-100,0])


# GENERATE MD DEFORMATIONS TORSIONS
bonds_nma, angles_nma, torsions_nma = cartesian_to_internal(nma_structure)
sim.plot_structure()

# torsions_nma[4] += -0.5
# torsions_nma[6] += 0.5
internal = np.array([bonds_nma[2:], angles_nma[1:], torsions_nma]).T
deformed_structure = internal_to_cartesian(internal, nma_structure[:3])
sim.plot_structure(deformed_structure)

sim.deformed_structure= deformed_structure

gaussian_sigma=2
sampling_rate=4
n_voxels=24
sim.compute_density(size=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate)
sim.plot_density()

########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

input_data = {
    # structure
             'n_atoms': init.n_atoms,
             'y': sim.deformed_structure,
             'epsilon': 1,
            'bonds': init.bonds[2:],
            'angles': init.angles[1:],
            'torsions': init.torsions,
            'first': init.coords[:3],


            'torsion_sigma': 0.001,
            'k_torsions': src.constants.K_TORSIONS,
            'n_torsions': src.constants.N_TORSIONS,
            'delta_torsions': src.constants.DELTA_TORSIONS,
            'k_U': src.constants.CARBON_MASS/(src.constants.K_BOLTZMANN * 1000),
            'R_sigma' : 0.001,
            'shift_sigma' :0.01,
            'max_shift': 0.1,
            'verbose':0,
    #modes
            'n_modes': init.modes.shape[1],
            'A_modes': init.modes,
            'sigma': 200,
    # EM density
             'N':sim.n_voxels,
             'halfN':int(sim.n_voxels/2),
             'gaussian_sigma':gaussian_sigma,
             'sampling_rate': sampling_rate,
             'em_density': sim.deformed_density,
             'epsilon_density': np.max(sim.deformed_density) / 10

}


fit =src.fitting.Fitting(input_data, "md_nma_torsions_emmap")
fit.optimizing(n_iter=1000)
# fit.sampling(n_chain=4, n_iter=100, n_warmup=800)
fit.plot_structure(save="results/sampling_structure_torsions_pas_nma.png")
# fit.plot_error_map(N=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate, slice=8)
fit.plot_nma(sim.q)
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
