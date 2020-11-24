import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting

########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
sim = src.simulation.Simulator.from_file(pdb_file="data/AK/AK.pdb", modes_file="data/AK/modes/vec.", n_modes=5)

sim.run_nma(amplitude=200)
nma_structure = np.array(sim.deformed_structure)
sim.run_md(U_lim=0.5, step=0.05, bonds={"k":0.001}, angles={"k":0.01}, lennard_jones={"k":1e-8, "d":3})
sim.plot_structure()

gaussian_sigma=2
sampling_rate=4
sim.compute_density(size=32, sigma=gaussian_sigma, sampling_rate=4)
sim.plot_density()

########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################


input_data = {'n_atoms': sim.n_atoms,
             'n_modes': sim.n_modes,
             'y': sim.deformed_structure,
             'x0': sim.init_structure,
             'A': sim.modes,
             'sigma':200,
             'epsilon':np.max(sim.deformed_density)/10,
             'mu':np.zeros(sim.n_modes),
             'U_init':sim.U_lim*2,
             's_md':sim.md_variance*2,
             'k_r':sim.bonds_k,
             'r0':sim.bonds_r0,
             'k_theta':sim.angles_k,
             'theta0':sim.angles_theta0,
             'k_lj':sim.lennard_jones_k,
             'd_lj':sim.lennard_jones_d,
             'N':sim.n_voxels,
             'halfN':int(sim.n_voxels/2),
             'gaussian_sigma':gaussian_sigma,
             'sampling_rate': sampling_rate,
             'em_density': sim.deformed_density
            }


fit = src.fitting.Fitting(input_data, "md_nma_emmap", build=True)
fit.sampling(n_iter=100, n_warmup=300, n_chain=4)
opt = fit.optimizing(n_iter=100)
vb = fit.vb(n_iter=1000)
fit.plot_nma(sim.q, save="results/modes_amplitudes.png")
fit.plot_structure(nma_structure, save="results/3d_structures.png")
fit.plot_error_map(N=sim.n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate, save="results/error_map.png")