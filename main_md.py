import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting

########################################################################################################
#               IMPORT FILES
########################################################################################################



# import PDB
atoms, ca = src.functions.read_pdb("data/AK/AK.pdb")
modes = src.functions.read_modes("data/AK/modes/vec.", n_modes=5)[ca][:]
atoms= src.functions.center_pdb(atoms)[ca][:]
sim = src.simulation.Simulator(atoms)

nma_structure = sim.run_nma(modes = modes, amplitude=200)
# rotated_structure = sim.rotate_pdb()
md_structure=  sim.run_md(U_lim=0.1, step=0.05, bonds={"k":0.001}, angles={"k":0.01}, lennard_jones={"k":1e-8, "d":3})
sim.plot_structure(nma_structure)

n_voxels=16
gaussian_sigma = 2
sampling_rate = 8
sim.compute_density(size=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate)
sim.plot_density()

########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

input_data = {
    # structure
             'n_atoms': sim.n_atoms,
             'n_modes': modes.shape[1],
             'y': sim.deformed_structure,
             'x0': sim.init_structure,
             'A': modes,
             'sigma':200,
             'epsilon': np.max(sim.deformed_density)/10,
             'mu':0,

    # Energy
             'U_init':0.1,
             's_md':8,
             'k_r':sim.bonds_k,
             'r0':sim.bonds_r0,
             'k_theta':sim.angles_k,
             'theta0':sim.angles_theta0,
             'k_lj':sim.lennard_jones_k,
             'd_lj':sim.lennard_jones_d,

    # EM density
             'N':sim.n_voxels,
             'halfN':int(sim.n_voxels/2),
             'gaussian_sigma':gaussian_sigma,
             'sampling_rate': sampling_rate,
             'em_density': sim.deformed_density
            }


fit =src.fitting.Fitting(input_data, "md_nma_emmap")
fit.sampling(n_iter=100, n_warmup=400, n_chain=4)
fit.plot_lp(save="results/sampling_lp.png")
fit.plot_nma(sim.q, save="results/sampling_nma.png")
fit.plot_structure(save="results/sampling_structure.png")
fit.plot_error_map(N=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate, save="results/sampling_err.png")

# fit_density = volume_from_pdb(fit.opt_results['x'], N=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate)
# rmse = root_mean_square_error(fit_density, sim.deformed_density)
# cc = cross_correlation(fit_density, sim.deformed_density)
#
# print("CC="+str(cc)+" ; RMSE="+str(rmse))
# print("Samling time = "+str(fit.sampling_time))
#
# fit_md = src.fitting.Fitting(input_data, "md_emmap")
# fit_md.optimizing(n_iter=10000)

