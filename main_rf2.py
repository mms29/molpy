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

target, target_ca = src.functions.read_pdb("data/ATPase/1su4_rotated.pdb")
# target, target_ca = src.functions.read_pdb("data/RF2/1n0u.pdb")
target= src.functions.center_pdb(target)[target_ca][::]

sim = src.simulation.Simulator(atoms)
sim.deformed_structure = target
sim.plot_structure()

modes = src.functions.read_modes("data/ATPase/modes/vec.", n_modes=5)[ca][:]

# nma_structure = sim.run_nma(modes = modes, amplitude=200)
md_structure=  sim.run_md(U_lim=1, step=0.05, bonds={"k":0.01}, angles={"k":0.001}, lennard_jones={"k":1e-8, "d":3})

gaussian_sigma=2
sampling_rate=6
sim.compute_density(size=32, sigma=gaussian_sigma, sampling_rate=sampling_rate)
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
             'sigma':50,
             'epsilon':np.max(sim.deformed_density)/100,
             'mu': 0,

    # Energy
             'U_init':1,
             's_md':5,
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


fit = src.fitting.Fitting(input_data, "md_emmap")
opt = fit.optimizing(n_iter=10000)
fit.plot_structure(save="results/atpase_md_structure.png")
fit.plot_error_map(N=sim.n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate, save="results/atpase_md_err.png")
fit.save("results/atpase_md_structure.pkl")