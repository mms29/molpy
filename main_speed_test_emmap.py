import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting

########################################################################################################
#               IMPORT FILES
########################################################################################################



# import PDB
atoms, ca = src.functions.read_pdb("data/AK/AK.pdb")
modes = src.functions.read_modes("data/AK/modes/vec.", n_modes=5)[ca]
atoms = src.functions.center_pdb(atoms)[ca]

sim = src.simulation.Simulator(atoms)
nma_structure = sim.run_nma(modes = modes, amplitude=[100,-50,150,-100,50])
md_structure=  sim.run_md(U_lim=0.01, step=0.01, bonds={"k":0.001}, angles={"k":0.01}, lennard_jones={"k":1e-8, "d":3})
sim.plot_structure(nma_structure)

n_voxels = 24
gaussian_sigma=2
sampling_rate= 128/n_voxels

N = 42
n_atoms = (np.arange(N)+1)*5

fit_md_opt_times         = np.zeros(N)
fit_md_error_density     = np.zeros(N)
fit_md_error_atoms       = np.zeros(N)
fit_md_cross_corr        = np.zeros(N)
fit_md_nma_opt_times     = np.zeros(N)
fit_md_nma_error_density = np.zeros(N)
fit_md_nma_error_atoms   = np.zeros(N)
fit_md_nma_cross_corr    = np.zeros(N)
fit_nma_opt_times        = np.zeros(N)
fit_nma_error_density    = np.zeros(N)
fit_nma_error_atoms      = np.zeros(N)
fit_nma_cross_corr       = np.zeros(N)


for i in range(N):
    print("///////////////////////////////////////////////////////////")
    print("///////////////////////////////////////////////////////////")
    print("///////////////////////////////////////////////////////////")
    print("ITER = "+str(i))
    print("///////////////////////////////////////////////////////////")
    print("///////////////////////////////////////////////////////////")
    print("///////////////////////////////////////////////////////////")
    deformed_structure = sim.deformed_structure[:n_atoms[i]]
    em_density = volume_from_pdb(deformed_structure, N=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate)
    input_data = {
        # structure
                 'n_atoms': n_atoms[i],
                 'n_modes': modes.shape[1],
                 'y': deformed_structure,
                 'x0': sim.init_structure[:n_atoms[i]],
                 'A': modes[:n_atoms[i]],
                 'sigma':200,
                 'epsilon': np.max(em_density)/10,
                 'mu':0,

        # Energy
                 'U_init':0.1,
                 's_md':5,
                 'k_r':sim.bonds_k,
                 'r0':sim.bonds_r0,
                 'k_theta':sim.angles_k,
                 'theta0':sim.angles_theta0,
                 'k_lj':sim.lennard_jones_k,
                 'd_lj':sim.lennard_jones_d,

        # EM density
                 'N':n_voxels,
                 'halfN':int(n_voxels/2),
                 'gaussian_sigma':gaussian_sigma,
                 'sampling_rate': sampling_rate,
                 'em_density':em_density
                }


    fit_nma = src.fitting.Fitting(input_data, "nma_emmap")
    fit_nma.optimizing(n_iter=10000)

    fit_md_nma = src.fitting.Fitting(input_data, "md_nma_emmap")
    fit_md_nma.optimizing(n_iter=10000)

    fit_md = src.fitting.Fitting(input_data, "md_emmap")
    fit_md.optimizing(n_iter=10000)

    fit_md_density = volume_from_pdb(fit_md.opt_results['x'], N=n_voxels, sigma=gaussian_sigma,
                                     sampling_rate=sampling_rate)
    fit_md_opt_times[i]     = fit_md.opt_time
    fit_md_error_density[i] = root_mean_square_error(fit_md_density, em_density)
    fit_md_cross_corr[i]    = cross_correlation(fit_md_density, em_density)
    fit_md_error_atoms[i]   = root_mean_square_error(fit_md.opt_results['x'], deformed_structure)

    fit_md_nma_density = volume_from_pdb(fit_md_nma.opt_results['x'], N=n_voxels, sigma=gaussian_sigma,
                                     sampling_rate=sampling_rate)
    fit_md_nma_opt_times[i] = fit_md_nma.opt_time
    fit_md_nma_error_density[i] = root_mean_square_error(fit_md_nma_density, em_density)
    fit_md_nma_cross_corr[i] = cross_correlation(fit_md_nma_density, em_density)
    fit_md_nma_error_atoms[i] = root_mean_square_error(fit_md_nma.opt_results['x'], deformed_structure)

    fit_nma_density = volume_from_pdb(fit_nma.opt_results['x'], N=n_voxels, sigma=gaussian_sigma,
                                     sampling_rate=sampling_rate)
    fit_nma_opt_times[i] = fit_nma.opt_time
    fit_nma_error_density[i] = root_mean_square_error(fit_nma_density, em_density)
    fit_nma_cross_corr[i] = cross_correlation(fit_nma_density, em_density)
    fit_nma_error_atoms[i] = root_mean_square_error(fit_nma.opt_results['x'], deformed_structure)

    #############################
    # Plots
    #################################

    fig, ax = plt.subplots(1, 4, figsize=(25, 8))

    ax[0].plot(n_atoms[:i + 1], fit_nma_opt_times[:i + 1])
    ax[0].plot(n_atoms[:i + 1], fit_md_nma_opt_times[:i + 1])
    ax[0].plot(n_atoms[:i + 1], fit_md_opt_times[:i + 1])
    ax[0].set_xlabel("number of atoms")
    ax[0].set_ylabel('Time (s)')
    ax[0].legend(["nma", "md_nma", "md"])
    ax[0].set_title("Time")

    ax[1].plot(n_atoms[:i + 1], fit_nma_error_density[:i + 1])
    ax[1].plot(n_atoms[:i + 1], fit_md_nma_error_density[:i + 1])
    ax[1].plot(n_atoms[:i + 1], fit_md_error_density[:i + 1])
    ax[1].set_xlabel("number of atoms")
    ax[1].set_ylabel('RMSE')
    ax[1].legend(["nma", "md_nma", "md"])
    ax[1].set_title("RMSE density")

    ax[2].plot(n_atoms[:i + 1], fit_nma_cross_corr[:i + 1])
    ax[2].plot(n_atoms[:i + 1], fit_md_nma_cross_corr[:i + 1])
    ax[2].plot(n_atoms[:i + 1], fit_md_cross_corr[:i + 1])
    ax[2].set_xlabel("number of atoms")
    ax[2].set_ylabel('CC')
    ax[2].legend(["nma", "md_nma", "md"])
    ax[2].set_title("CC")

    ax[3].plot(n_atoms[:i + 1], fit_nma_error_atoms[:i + 1])
    ax[3].plot(n_atoms[:i + 1], fit_md_nma_error_atoms[:i + 1])
    ax[3].plot(n_atoms[:i + 1], fit_md_error_atoms[:i + 1])
    ax[3].set_xlabel("number of atoms")
    ax[3].set_ylabel('RMSE')
    ax[3].legend(["nma", "md_nma", "md"])
    ax[3].set_title("RMSE atoms")

    fig.savefig("results/speed_emmap.png")
    with open("results/speed_emmap.pkl", 'wb') as f:
        data = {
            "fit_md_opt_times": fit_md_opt_times,
            "fit_md_error_density": fit_md_error_density,
            "fit_md_error_atoms": fit_md_error_atoms,
            "fit_md_cross_corr": fit_md_cross_corr,
            "fit_md_nma_opt_times": fit_md_nma_opt_times,
            "fit_md_nma_error_density": fit_md_nma_error_density,
            "fit_md_nma_error_atoms": fit_md_nma_error_atoms,
            "fit_md_nma_cross_corr": fit_md_nma_cross_corr,
            "fit_nma_opt_times": fit_nma_opt_times,
            "fit_nma_error_density": fit_nma_error_density,
            "fit_nma_error_atoms": fit_nma_error_atoms,
            "fit_nma_cross_corr": fit_nma_cross_corr
        }
        pickle.dump(obj=data, file=f)