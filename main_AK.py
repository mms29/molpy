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

########################################################################################################
#               IMAGES
########################################################################################################

import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting
from src.viewers import structures_viewer
from src.flexible_fitting import *
import src.molecule
import src.io
import copy


# import PDB
init =src.io.read_pdb("data/AK/AK.pdb")
init.center_structure()
init.add_modes("data/AK/modes/vec.", n_modes=4)
init = init.select_atoms(pattern='CA')
init.rotate([np.pi/2,np.pi/2,0])


sim = src.simulation.Simulator(init)
nma = sim.nma_deform([300,-100,0,0])
target = nma.energy_min(U_lim=10000, step=0.005)
target.coords = copy.copy(init.coords)
target.rotate([np.pi/2,0,0])
size=64
sampling_rate=2
gaussian_sigma=2
threshold=5
target_img = target.to_image(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold)
init_img = init.to_image(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold)
target_img.show()
init_img.show()

coord=init.coords
psim =init_img.data
pexp =target_img.data
sigma=gaussian_sigma
A_modes=init.modes

params ={
    "x_init" : np.zeros(init.coords.shape),
    "q_init" : np.zeros(init.modes.shape[1]),
    "angles_init" : np.zeros(3),

    "lb" : 100,#CARBON_MASS /(K_BOLTZMANN*T *3*init.n_atoms) *200,
    "lp" : 1,#CARBON_MASS /(K_BOLTZMANN*T *3*init.n_atoms),
    "lx" : 0,
    "lq" : 0,

    "max_iter": 10,
    "criterion" :False,

    "dxt" : 0.015,
    "dqt" : 0.1,
    "danglest": 0.001,

    "m_vt" : 1,#np.sqrt(K_BOLTZMANN*T /CARBON_MASS),
    "m_wt" : 10,
    "m_anglest" : 0.01,
}
n_iter=10
n_warmup = 5

fit1  =FlexibleFitting(init, target_img)
fit1.HMC(mode="ROT", params=params, n_iter=n_iter, n_warmup=n_warmup, verbose=False)
fit1.show()
structures_viewer([target, fit1.res["mol"]])
