
import matplotlib.pyplot as plt
import src.simulation
from src.viewers import structures_viewer, chimera_structure_viewer
from src.flexible_fitting import *
import src.io
from src.constants import *
########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
init =src.io.read_pdb("data/AK/AK_PSF.pdb")
init.center_structure()
init.add_modes("data/AK/modes_psf/vec.", n_modes=4)
# init.rotate([0,np.pi/2,0])
#

# init.set_forcefield(psf_file="data/AK/AK.psf")
init.select_atoms(pattern='CA')
init.set_forcefield()

sim = src.simulation.Simulator(init)
q = [300,-100,0,0]
target = sim.nma_deform(q)
structures_viewer([target, init])

size=64
sampling_rate=2.2
threshold= 4
gaussian_sigma=2
target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold)
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold)
target_density.show()

########################################################################################################
#               HMC
########################################################################################################
T=1000

params ={
    "x_init" : np.zeros(init.coords.shape),
    "q_init" : np.zeros(init.modes.shape[1]),

    "lb" : 100, #np.mean(init.prm.mass)*ATOMIC_MASS_UNIT /(K_BOLTZMANN*T *3*init.n_atoms) *1e6,
    "lp" : 1, #np.mean(init.prm.mass)*ATOMIC_MASS_UNIT /(K_BOLTZMANN*T *3*init.n_atoms)*1e6,

    "max_iter": 20,
    "criterion" :True,

    "x_dt" : 0.01,
    "q_dt" : 0.1,

    "x_mass" :  1,#np.sqrt(K_BOLTZMANN*T /(np.mean(init.prm.mass)*ATOMIC_MASS_UNIT)),
    # np.array([np.sqrt(K_BOLTZMANN*T /(init.prm.mass*ATOMIC_MASS_UNIT)),
    #           np.sqrt(K_BOLTZMANN*T /(init.prm.mass*ATOMIC_MASS_UNIT)),
    #           np.sqrt(K_BOLTZMANN*T /(init.prm.mass*ATOMIC_MASS_UNIT))]),
    "q_mass" : 1,
}
n_iter=20
n_warmup = n_iter // 2


fit  =FlexibleFitting(init = init, target= target_density, mode="HMC", params=params,
                      n_iter=n_iter, n_warmup=n_warmup, n_chain=4, verbose=2)

fit.HMC()
fit.show()
fit.show_3D()

chimera_structure_viewer([fit.res["mol"], target])

########################################################################################################
#               IMAGES
########################################################################################################
#
# import matplotlib.pyplot as plt
# from src.functions import *
# import src.simulation
# import src.fitting
# from src.viewers import structures_viewer
# from src.flexible_fitting import *
# import src.molecule
# import src.io
# import copy
#
#
# # import PDB
# init =src.io.read_pdb("data/AK/AK.pdb")
# init.center_structure()
# init.add_modes("data/AK/modes/vec.", n_modes=4)
# init = init.select_atoms(pattern='CA')
# init.rotate([np.pi/2,np.pi/2,0])
#
#
# sim = src.simulation.Simulator(init)
# nma = sim.nma_deform([300,-100,0,0])
# target = nma.energy_min(U_lim=10000, step=0.005)
# target.coords = copy.copy(init.coords)
# target.rotate([np.pi/2,0,0])
# size=64
# sampling_rate=2
# gaussian_sigma=2
# threshold=5
# target_img = target.to_image(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold)
# init_img = init.to_image(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold)
# target_img.show()
# init_img.show()
#
# coord=init.coords
# psim =init_img.data
# pexp =target_img.data
# sigma=gaussian_sigma
# A_modes=init.modes
#
# params ={
#     "x_init" : np.zeros(init.coords.shape),
#     "q_init" : np.zeros(init.modes.shape[1]),
#     "angles_init" : np.zeros(3),
#
#     "lb" : 100,#CARBON_MASS /(K_BOLTZMANN*T *3*init.n_atoms) *200,
#     "lp" : 1,#CARBON_MASS /(K_BOLTZMANN*T *3*init.n_atoms),
#     "lx" : 0,
#     "lq" : 0,
#
#     "max_iter": 10,
#     "criterion" :False,
#
#     "dxt" : 0.015,
#     "dqt" : 0.1,
#     "danglest": 0.001,
#
#     "m_vt" : 1,#np.sqrt(K_BOLTZMANN*T /CARBON_MASS),
#     "m_wt" : 10,
#     "m_anglest" : 0.01,
# }
# n_iter=10
# n_warmup = 5
#
# fit1  =FlexibleFitting(init, target_img)
# fit1.HMC(mode="ROT", params=params, n_iter=n_iter, n_warmup=n_warmup, verbose=False)
# fit1.show()
# structures_viewer([target, fit1.res["mol"]])
