import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting
import pickle
########################################################################################################
#               IMPORT FILES
########################################################################################################



# import PDB
atoms, ca = src.functions.read_pdb("data/AK/AK.pdb")
modes = src.functions.read_modes("data/AK/modes/vec.", n_modes=5)[ca]
atoms= src.functions.center_pdb(atoms)[ca]
sim = src.simulation.Simulator(atoms)

n_per_value = 10
N =40
param = np.array([ round(0.1*(1.2**i),6) for i in range(N)])
param_name= "md_variance"
print(param)

fit_md_opt_times         = np.zeros((N, n_per_value))
fit_md_error_density     = np.zeros((N, n_per_value))
fit_md_error_atoms       = np.zeros((N, n_per_value))
fit_md_cross_corr        = np.zeros((N, n_per_value))

fit_md_nma_opt_times     = np.zeros((N, n_per_value))
fit_md_nma_error_density = np.zeros((N, n_per_value))
fit_md_nma_error_atoms   = np.zeros((N, n_per_value))
fit_md_nma_cross_corr    = np.zeros((N, n_per_value))

########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

for i in range(N):
    nma_structure = sim.run_nma(modes=modes, amplitude=150)
    md_structure = sim.run_md(U_lim=0.1, step=0.01, bonds={"k": 0.001}, angles={"k": 0.01},
                              lennard_jones={"k": 1e-8, "d": 3})

    N = 16
    gaussian_sigma = 2
    sampling_rate = 8
    sim.compute_density(size=N, sigma=gaussian_sigma, sampling_rate=sampling_rate)

    for j in range(n_per_value):

        input_data = {
            # structure
                     'n_atoms': sim.n_atoms,
                     'n_modes': modes.shape[1],
                     'y': sim.deformed_structure,
                     'x0': sim.init_structure,
                     'A': modes,
                     'sigma':150,
                     'epsilon': np.max(sim.deformed_density)/10,
                     'mu':0,

            # Energy
                     'U_init':0.1,
                     's_md':param[i],
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


        fit_md_nma = src.fitting.Fitting(input_data, "md_nma_emmap")
        fit_md_nma.optimizing(n_iter=10000)

        fit_md = src.fitting.Fitting(input_data, "md_emmap")
        fit_md.optimizing(n_iter=10000)

        fit_md_density = volume_from_pdb(fit_md.opt_results['x'], N=N, sigma=gaussian_sigma, sampling_rate=sampling_rate)
        fit_md_opt_times     [i,j] = fit_md.opt_time
        fit_md_error_density [i,j] = root_mean_square_error(fit_md_density, sim.deformed_density)
        fit_md_cross_corr    [i,j] = cross_correlation(fit_md_density, sim.deformed_density)
        fit_md_error_atoms   [i,j] = root_mean_square_error(fit_md.opt_results['x'], sim.deformed_structure)

        fit_md_nma_density = volume_from_pdb(fit_md_nma.opt_results['x'], N=N, sigma=gaussian_sigma, sampling_rate=sampling_rate)
        fit_md_nma_opt_times     [i,j] = fit_md_nma.opt_time
        fit_md_nma_error_density [i,j] = root_mean_square_error(fit_md_nma_density, sim.deformed_density)
        fit_md_nma_cross_corr    [i,j] = cross_correlation(fit_md_nma_density, sim.deformed_density)
        fit_md_nma_error_atoms   [i,j] = root_mean_square_error(fit_md_nma.opt_results['x'], sim.deformed_structure)

        #############################
        #Plots
        #################################

        fig, ax = plt.subplots(1, 4, figsize=(25, 8))

        ax[0].plot(param[:i + 1], np.mean(fit_md_nma_opt_times[:i + 1, :j + 1], axis=1))
        ax[0].plot(param[:i + 1], np.mean(fit_md_opt_times[:i + 1, :j + 1], axis=1))
        ax[0].set_xlabel(param_name)
        ax[0].set_ylabel('Time (s)')
        ax[0].legend(["md_nma", "md"])
        ax[0].set_title("Time")

        ax[1].plot(param[:i + 1], np.mean(fit_md_nma_error_density[:i + 1, :j + 1], axis=1))
        ax[1].plot(param[:i + 1], np.mean(fit_md_error_density[:i + 1, :j + 1], axis=1))
        ax[1].set_xlabel(param_name)
        ax[1].set_ylabel('RMSE')
        ax[1].legend(["md_nma", "md"])
        ax[1].set_title("RMSE density")

        ax[2].plot(param[:i + 1], np.mean(fit_md_nma_cross_corr[:i + 1, :j + 1], axis=1))
        ax[2].plot(param[:i + 1], np.mean(fit_md_cross_corr[:i + 1, :j + 1], axis=1))
        ax[2].set_xlabel(param_name)
        ax[2].set_ylabel('CC')
        ax[2].legend(["md_nma", "md"])
        ax[2].set_title("CC")

        ax[3].plot(param[:i + 1], np.mean(fit_md_nma_error_atoms[:i + 1, :j + 1], axis=1))
        ax[3].plot(param[:i + 1], np.mean(fit_md_error_atoms[:i + 1, :j + 1], axis=1))
        ax[3].set_xlabel(param_name)
        ax[3].set_ylabel('RMSE')
        ax[3].legend(["md_nma", "md"])
        ax[3].set_title("RMSE atoms")

        fig.savefig("results/md_variance_parameter_test.png")

        with open("results/md_variance_parameter_test.pkl", 'wb') as f:
            data={
            "fit_md_opt_times": fit_md_opt_times         ,
            "fit_md_error_density": fit_md_error_density     ,
            "fit_md_error_atoms": fit_md_error_atoms       ,
            "fit_md_cross_corr": fit_md_cross_corr        ,
            "fit_md_nma_opt_times": fit_md_nma_opt_times     ,
            "fit_md_nma_error_density": fit_md_nma_error_density ,
            "fit_md_nma_error_atoms": fit_md_nma_error_atoms   ,
            "fit_md_nma_cross_corr": fit_md_nma_cross_corr
            }
            pickle.dump(obj =data, file=f)
