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

nma_structure = sim.run_nma(modes = modes, amplitude=150)
sim.plot_structure()


########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

input_data = {
    # structure
             'n_atoms_x': sim.n_atoms,
             'n_atoms_y': sim.n_atoms,
             'n_modes': modes.shape[1],
             'y': sim.deformed_structure,
             'x0': sim.init_structure,
             'A': modes,
             'sigma':150,
             'epsilon': 1,
             'mu':0,
             'threshold':1
           }


fit =src.fitting.Fitting(input_data, "pseudo-atomic")
# fit.optimizing(n_iter=10000)
fit.sampling(n_iter=25, n_warmup=100, n_chain=4)
fit.plot_structure(save="results/sampling_structure.png")
fit.plot_lp(save="results/sampling_lp.png")
fit.plot_nma(sim.q, save="results/sampling_nma.png")
