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
init =src.io.read_pdb("data/AK/AK.pdb")
init.center_structure()
init.add_modes("data/AK/modes/vec.", n_modes=4)
init = init.select_atoms(pattern='CA')


sim = src.simulation.Simulator(init)
nma = sim.nma_deform([300,-100,0,0])
target = sim.mc_deform()
structures_viewer([target, init, nma])


size=64
sampling_rate=3
threshold= round(4/sampling_rate *2)
gaussian_sigma=2
target_density = target.to_density(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold)
init_density = init.to_density(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold)
target_density.show()

########################################################################################################
#               HMC
########################################################################################################


mol, q_res = HNMA(init=init, q_init = None, target_density=target_density, n_iter = 20, n_warmup=10, k=1, dt=5)

mol, x_res, q_res = HMCNMA(init=init, q_init = None, target_density=target_density, n_iter = 20,n_warmup=10, max_iter=1000, k=100,  dxt=0.02, dqt=0.1)


mol, x_res = HMC(init=init, target_density=target_density, n_iter = 20, n_warmup=10, k=10, dt=0.02, size=size, sigma=gaussian_sigma,
                          sampling_rate=sampling_rate, threshold=threshold)

src.viewers.chimera_fit_viewer(mol, target_density, genfile="data/AK/AK.pdb")


########################################################################################################
#               OPT
########################################################################################################

t = time.time()

x_res1, q_res1 = NMA_gradient_descend(init=init, target_density=target_density,n_iter=10, dt=10,
                                   size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=threshold)
x =src.molecule.Molecule(x_res)
x_density = x.to_density(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold)

cross_correlation(x_density.data, target_density.data)
print(t-time.time())
plot_structure([target.coords, x_res1, init.coords, nma.coords], ["target", "res", "init", "nma"])

t = time.time()

x_res2, q_res2 = MCNMA_gradient_descend(init=init, target_density=target_density,n_iter=30, dxt=0.001,
                                    dqt=10, k = 100,
                                   size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=threshold)
print(t-time.time())

t = time.time()

x_res3 = MC_gradient_descend(init=init, target_density=target_density,n_iter=100, dt=0.001, k=200,
                                   size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=threshold)
print(t-time.time())

# plot_structure([target.coords, x_res3], ["target", "res", "init"])
#
# fitted = src.molecule.Molecule(x_res3, chain_id=init.chain_id)
# src.io.save_pdb(fitted, file = "results/P97/fitted_MC.pdb", gen_file="data/P97/5ftm.pdb")
# src.io.save_pdb(init, file = "results/P97/init.pdb", gen_file="data/P97/5ftm.pdb")
# src.io.save_pdb(test, file = "results/P97/test.pdb", gen_file="data/P97/5ftm.pdb")
# src.io.save_density(target_density,outfilename= "results/P97/target.mrc")
#
