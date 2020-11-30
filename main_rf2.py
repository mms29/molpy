import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting

########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
atoms, ca = src.functions.read_pdb("data/ATPase/1iwo.pdb")
# atoms, ca = src.functions.read_pdb("data/RF2/1n0v.pdb")
atoms= src.functions.center_pdb(atoms)[ca][::]

target, target_ca = src.functions.read_pdb("data/ATPase/1su4.pdb")
# target, target_ca = src.functions.read_pdb("data/RF2/1n0u.pdb")
target= src.functions.center_pdb(target)[target_ca][::]

sim = src.simulation.Simulator(atoms)
sim.deformed_structure = target
sim.plot_structure()

modes = src.functions.read_modes("data/ATPase/modes/vec.", n_modes=5)[ca][:]

# nma_structure = sim.run_nma(modes = modes, amplitude=200)
md_structure=  sim.run_md(U_lim=1, step=0.05, bonds={"k":0.01}, angles={"k":0.001}, lennard_jones={"k":1e-8, "d":3})

gaussian_sigma=2
sampling_rate=8
sim.compute_density(size=24, sigma=gaussian_sigma, sampling_rate=sampling_rate)
sim.plot_density()

########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

# n_shards=2
# os.environ['STAN_NUM_THREADS'] = str(n_shards)

input_data = {
    # structure
             'n_atoms': sim.n_atoms,
             'n_modes': modes.shape[1],
             'y': sim.deformed_structure,
             'x0': sim.init_structure,
             'A': modes,
             'sigma':100,
             'epsilon':np.max(sim.deformed_density)/10,
             'mu': 0,
             # 'n_shards':n_shards,

    # Energy
             'U_init':sim.U_lim,
             's_md':20,
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


fit = src.fitting.Fitting(input_data, "md_nma_emmap_rotation")
# fit.sampling(n_iter=50, n_warmup=200, n_chain=4)
opt = fit.optimizing(n_iter=1000)
# vb = fit.vb(n_iter=100)
# fit.plot_nma(save="results/modes_amplitudes.png")
fit.plot_structure(save="results/3d_structures.png")
fit.plot_error_map(N=sim.n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate, save="results/error_map.png")