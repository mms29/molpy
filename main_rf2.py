import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting
from src.viewers import structures_viewer
from src.flexible_fitting import *
import src.molecule
import src.io

########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
init =src.io.read_pdb("data/ATPase/1iwo.pdb")
init.add_modes("data/ATPase/modes/vec.", n_modes=10)
init.center_structure()
init = init.select_atoms(pattern='CA')


target =src.io.read_pdb("data/ATPase/1su4_rotated3.pdb")
target.center_structure()
target = target.select_atoms(pattern='CA')

structures_viewer([init, target])

size = 128
gaussian_sigma=2
sampling_rate = 2
threshold = 4
target_density = target.to_density( size=size, gaussian_sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=threshold)
target_density.show()
########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################


x_res = HMC(init=init, target_density=target_density, n_iter = 10, k=10, dt=0.03, size=size, sigma=gaussian_sigma,
                          sampling_rate=sampling_rate, threshold=threshold)

x_res, q_res = HMCNMA(init=init, target_density=target_density, n_iter = 10, k=10, dxt=0.03, kq=1, dqt=5, size=size, sigma=gaussian_sigma,
                          sampling_rate=sampling_rate, threshold=threshold)

q_res = HNMA(init=init, target_density=target_density, n_iter = 50, k=1, dt=5, size=size, sigma=gaussian_sigma,
                          sampling_rate=sampling_rate, threshold=threshold)

coord = init.coords  + np.dot(np.mean(q_res[30:], axis=0), init.modes)+ np.mean(x_res[8:], axis=0)
plot_structure([target.coords, coord, init.coords], ["target", "res", "init"])

fitted = src.molecule.Molecule(coord, chain_id=init.chain_id)
fitted.get_energy()
src.viewers.chimera_fit_viewer(fitted, target_density, genfile="data/ATPase/1iwo.pdb")

test = fitted.to_density(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold)
cross_correlation(target_density.data, test.data)