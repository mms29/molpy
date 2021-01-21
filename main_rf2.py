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
init = init.select_atoms(pattern='CA')


target =src.io.read_pdb("data/ATPase/1su4_rotated3.pdb")
target = target.select_atoms(pattern='CA')

structures_viewer([init, target])

size = 64
gaussian_sigma=2
sampling_rate = 4
threshold = 4
target_density = src.molecule.Density(volume_from_pdb_fast3(target.coords, size=size, sigma=gaussian_sigma,
                    sampling_rate=sampling_rate, threshold=threshold), sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma)
target_density.show()
########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

res = gradient_descend(init, target_density.data, n_iter =50000,k =10,dt=0.001,size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=threshold)

x_res, q_res = NMA_gradient_descend(init, target_density.data, n_iter =1000,dt=1,size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=threshold)
x_res, q_res = MCNMA_gradient_descend(init, target_density.data, n_iter =10000,dqt=10, dxt=0.001,k=100,
                                      size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=threshold)

plot_structure([init.coords, x_res, target.coords], ["init", "fit", "target"])

fitted = src.molecule.Molecule.from_coords(x_res)
src.io.save_pdb(fitted, file = "results/ATPase/1iwo_fitted_MCNMA.pdb", gen_file="data/ATPase/1iwo.pdb")
src.io.save_pdb(init, file = "results/ATPase/init.pdb", gen_file="data/ATPase/1iwo.pdb")
src.io.save_density(target_density,outfilename= "results/ATPase/target.mrc")
# fit = src.fitting.Fitting(input_data, "md_torsions")
# opt = fit.optimizing(n_iter=10000)
# fit.plot_structure(save="results/atpase_md_structure.png")
# fit.plot_error_map(N=32, sigma=gaussian_sigma, sampling_rate=sampling_rate, save="results/atpase_md_err.png")
# fit.save("results/atpase_md_structure.pkl")
#
# fit = src.fitting.Fitting.load("results/atpase_md_nma_structure.pkl")
# fit .plot_structure()