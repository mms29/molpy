from src.functions import *
import src.simulation
import src.fitting
import src.constants
import src.molecule
import src.io
import matplotlib.pyplot as plt

# import PDB
atoms, ca = src.functions.read_pdb("data/AK/AK.pdb")
modes = src.functions.read_modes("data/AK/modes/vec.", n_modes=3)[ca][:]
atoms= src.functions.center_pdb(atoms)[ca][:]

sim = src.simulation.Simulator(atoms)


#GENERATE NMA DEFORMATIONS
nma_structure = sim.run_nma(modes = modes, amplitude=[200,-100,0])


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

bonds, angles, torsions = cartesian_to_internal(atoms)

########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

input_data = {
    # structure
             'n_atoms': atoms.shape[0],
             'y': sim.deformed_structure,
             'epsilon': 1,
            'bonds': bonds[2:],
            'angles': angles[1:],
            'torsions': torsions,
            'first': atoms[:3],
            'first_sigma' : 0.1,
            'first_max': 1,
            'torsion_sigma': 1,
            'torsion_max': 10,
            'k_torsions': src.constants.K_TORSIONS,
            'n_torsions': src.constants.N_TORSIONS,
            'delta_torsions': src.constants.DELTA_TORSIONS,
            'k_U': src.constants.CARBON_MASS/(src.constants.K_BOLTZMANN * 1000),
            'R_sigma' : 0.1,
            'shift_sigma' :1,
            'max_shift': 10,
            'verbose':0,
    #modes
            # 'n_modes': modes.shape[1],
            # 'A': modes,
            # 'q_sigma': 50,
            # 'q_mu': 0,
            # 'q_max': 200,
    # EM density
             'N':sim.n_voxels,
             'halfN':int(sim.n_voxels/2),
             'gaussian_sigma':gaussian_sigma,
             'sampling_rate': sampling_rate,
             'em_density': sim.deformed_density,
             'epsilon_density': np.max(sim.deformed_density) / 10

}


fit =src.fitting.Fitting(input_data, "md_torsions")
fit.optimizing(n_iter=25000)
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
