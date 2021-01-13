import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting

########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
atoms, ca = read_pdb("data/ATPase/1iwo.pdb")
# atoms, ca = src.functions.read_pdb("data/RF2/1n0v.pdb")
atoms= center_pdb(atoms)[ca][::]
compute_u_init(atoms, bonds={"k":0.01}, angles={"k":0.001}, lennard_jones={"k":1e-8, "d":3})


target, target_ca = read_pdb("data/ATPase/1su4_rotated.pdb")
# target, target_ca = src.functions.read_pdb("data/RF2/1n0u.pdb")
target= center_pdb(target)[target_ca][::]
compute_u_init(target, bonds={"k":0.01}, angles={"k":0.001}, lennard_jones={"k":1e-8, "d":3})


sim = src.simulation.Simulator(atoms)
sim.deformed_structure = target
sim.plot_structure()

modes = read_modes("data/ATPase/modes/vec.", n_modes=13)[ca][:]

# nma_structure = sim.run_nma(modes = modes, amplitude=200)
md_structure=  sim.run_md(U_lim=1, step=0.05, bonds={"k":0.01}, angles={"k":0.001}, lennard_jones={"k":1e-8, "d":3})

gaussian_sigma=1
sampling_rate=8
sim.compute_density(size=24, sigma=gaussian_sigma, sampling_rate=sampling_rate)
sim.plot_density()

bonds, angles, torsions = cartesian_to_internal(atoms)
bonds2, angles2, torsions2 = cartesian_to_internal(target)

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
             'sigma':100,
             'epsilon':np.max(sim.deformed_density)/50,
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
             'em_density': sim.deformed_density,

    'bonds': bonds[2:],
    'angles': angles[1:],
    'torsions': torsions,
    'first': atoms[:3],
    'epsilon': 1,#np.max(sim.deformed_density) / 50,  # 1
    'first_sigma': 10,
    'first_max': 25,
    'torsion_sigma': 0.1,
    'torsion_max': 1,
    'angle_sigma' : 0.01,
    'angle_max': 0.5,
    'bond_sigma' : 0.01,
    'bond_max' : 1,
            }



fit = src.fitting.Fitting(input_data, "md_torsions")
opt = fit.optimizing(n_iter=10000)
fit.plot_structure(save="results/atpase_md_structure.png")
fit.plot_error_map(N=32, sigma=gaussian_sigma, sampling_rate=sampling_rate, save="results/atpase_md_err.png")
fit.save("results/atpase_md_structure.pkl")

fit = src.fitting.Fitting.load("results/atpase_md_nma_structure.pkl")
fit .plot_structure()