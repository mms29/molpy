import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting

########################################################################################################
#               IMPORT FILES
########################################################################################################



# import PDB
atoms, ca = src.functions.read_pdb("data/AK/AK.pdb")
modes = src.functions.read_modes("data/AK/modes/vec.", n_modes=3)[ca][:]
atoms= src.functions.center_pdb(atoms)[ca][:]
sim = src.simulation.Simulator(atoms)

nma_structure = sim.run_nma(modes = modes, amplitude=[-150,0, 100])
# rotated_structure = sim.rotate_pdb()
md_structure=  sim.run_md(U_lim=0.05, step=0.01, bonds={"k":0.001}, angles={"k":0.01}, lennard_jones={"k":1e-8, "d":3})
sim.plot_structure(nma_structure)
compute_u_init(atoms, bonds={"k":0.001}, angles={"k":0.01}, lennard_jones={"k":1e-8, "d":3})
print('///')
compute_u_init(nma_structure, bonds={"k":0.001}, angles={"k":0.01}, lennard_jones={"k":1e-8, "d":3})
print('///')
compute_u_init(md_structure, bonds={"k":0.001}, angles={"k":0.01}, lennard_jones={"k":1e-8, "d":3})

n_voxels=32
gaussian_sigma = 2
sampling_rate = 4
sim.compute_density(size=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate)
# sim.plot_density()

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
             'epsilon': np.max(sim.deformed_density)/50,
             'mu':0,
             'q_max':200,

    # Energy
             'U_init':0.01,
             's_md':0.2,
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

print(root_mean_square_error(sim.deformed_density, sim.init_density))

fit1 =src.fitting.Fitting(input_data, "nma_emmap")
fit1.optimizing(n_iter=10000)
fit1.plot_structure(save="results/sampling_structure.png")
fit1.plot_error_map(N=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate, save="results/sampling_err.png")
fit1.plot_nma(sim.q, save="results/sampling_nma.png")
opt_density = volume_from_pdb(fit1.opt_results['x'],N=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate)
print(cross_correlation(sim.deformed_density, opt_density))
print(root_mean_square_error(sim.deformed_density, opt_density))

fit2 =src.fitting.Fitting(input_data, "md_nma_emmap")
fit2.optimizing(n_iter=10000)
fit2.plot_structure(save="results/sampling_structure.png")
fit2.plot_error_map(N=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate, save="results/sampling_err.png")
fit2.plot_nma(sim.q, save="results/sampling_nma.png")
opt_density = volume_from_pdb(fit2.opt_results['x'],N=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate)
print(cross_correlation(sim.deformed_density, opt_density))
print(root_mean_square_error(sim.deformed_density, opt_density))
print(root_mean_square_error(sim.deformed_structure, fit2.opt_results['x']))


fit3 =src.fitting.Fitting(input_data, "md_emmap")
fit3.optimizing(n_iter=10000)
fit3.plot_structure(save="results/sampling_structure.png")
fit3.plot_error_map(N=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate, save="results/sampling_err.png")
opt_density = volume_from_pdb(fit3.opt_results['x'],N=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate)
print(cross_correlation(sim.deformed_density, opt_density))
print(root_mean_square_error(sim.deformed_density, opt_density))
print(root_mean_square_error(sim.deformed_structure, fit3.opt_results['x']))

compute_u_init(fit3.opt_results['x'], bonds={"k":0.001}, angles={"k":0.01}, lennard_jones={"k":1e-8, "d":3})

# fit_density = volume_from_pdb(fit.opt_results['x'], N=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate)
# rmse = root_mean_square_error(fit_density, sim.deformed_density)
# cc = cross_correlation(fit_density, sim.deformed_density)
#
# print("CC="+str(cc)+" ; RMSE="+str(rmse))
# print("Samling time = "+str(fit.sampling_time))
#
# fit_md = src.fitting.Fitting(input_data, "md_emmap")
# fit_md.optimizing(n_iter=10000)

