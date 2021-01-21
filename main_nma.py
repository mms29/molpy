import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting
import src.constants
import src.molecule
import src.io
import src.viewers
import matplotlib.pyplot as plt

########################################################################################################
#               IMPORT FILES
########################################################################################################


# import PDB
init =src.io.read_pdb("data/AK/AK.pdb")
init.add_modes("data/AK/modes/vec.", n_modes=3)
init = init.select_atoms(pattern='CA')
init.rotate(angles = [0,np.pi/2, np.pi/2])

sim = src.simulation.Simulator(init.coords)

#GENERATE NMA DEFORMATIONS
nma_structure = sim.run_nma(modes = init.modes, amplitude=[300,-100,0])


target = src.molecule.Molecule.from_coords(nma_structure)





# target.rotate(angles = [0,np.pi/2, np.pi/2])
n_pixels=128
sampling_rate=0.75
gaussian_sigma=4
target_image = target.to_image(n_pixels,gaussian_sigma,sampling_rate)
target_image.show()

########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

input_data = {
    # structure
             'n_atoms': target.n_atoms,
             'n_modes': init.modes.shape[1],
             'y': target.coords,
             'x0': init.coords,
             'A': init.modes,
             'sigma':400,
             'epsilon': np.max(target_image.data)/10,
             'mu':0,
            'image': target_image.data,
            'N' :n_pixels,
            'sampling_rate': sampling_rate,
            'gaussian_sigma' :gaussian_sigma,
            'halfN': int(n_pixels/2)
           }


fit =src.fitting.Fitting(input_data, "md")
fit.optimizing(n_iter=10000)
fit.sampling(n_iter=25, n_warmup=100, n_chain=4)
fit.plot_structure(save="results/sampling_structure.png")
fit.plot_lp(save="results/sampling_lp.png")
fit.plot_nma(sim.q, save="results/sampling_nma.png")
