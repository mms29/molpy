import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting
from src.viewers import structures_viewer, chimera_structure_viewer
from src.flexible_fitting import *
import src.molecule
import src.io
import pickle
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('number',  type=int)
args = parser.parse_args()
number = args.number

print("NUMBER=="+str(number))
########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
init =src.io.read_pdb("data/P97/5ftm.pdb")
init.center_structure()
# init.rotate([0,np.pi/2,0])
init.add_modes("data/P97/modes/vec.", n_modes=43)
# init.select_modes(np.array([10, 13, 28,35,42,44,47,49])-7)
init.select_modes(np.array([10, 13, 28])-7)
init = init.select_atoms(pattern='CA')

# sim = src.simulation.Simulator(init)
# target = sim.nma_deform([1000,0,0,-1000, -1000, -1000, 0, 500])
# target = nma.energy_min(500000, 0.01)
# nma = sim.nma_deform([1000,0,0,0, 0, 0, 0, 0])
# target = nma.energy_min(500000, 0.01)
#
# i =init.chain_id[1]
# coords = np.array(sim.deformed_mol[-1].coords)
# coords[0:i] += np.dot(np.array([2000,0,0,0,0,0,0,0]), init.modes[0:i])
# target =src.molecule.Molecule(coords = coords,chain_id=init.chain_id)

# # target = target.energy_min(U_lim=210000, step=0.01)
#
# chimera_structure_viewer([target, init], genfile="data/P97/5ftm.pdb")
# structures_viewer([target, init])
#
target =src.io.read_pdb("data/P97/5ftn.pdb")
target.center_structure()
target = target.select_atoms(pattern='CA')
# target.rotate([0,np.pi/2,0])

# structures_viewer([test, init, target])


size=128
sampling_rate=2
threshold=4
gaussian_sigma=2
target_density = target.to_density(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold)
target_density.show()

########################################################################################################
#               HMC
########################################################################################################

mol1, q_res1, l1, cc1 = HNMA(init=init, q_init = None, target_density=target_density, n_iter = 20, n_warmup=10, k=1, dt=0.4)

mol2, x_res2, q_res2, l2, cc2= HMCNMA(init=init, q_init = None, target_density=target_density, n_iter = 20, n_warmup=10, max_iter=50, k=100, dxt=0.005, dqt=0.4, m_test=10)

mol3, x_res3, l3, cc3 = HMC(init=init, target_density=target_density, n_iter = 20, n_warmup=10, k=100, dt=0.005, max_iter=50)


cc_init = cross_correlation(init.to_density(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold).data, target_density.data)
CC1=np.array([cc_init] + cc1)
CC2=np.array([cc_init] + cc2)
CC3=np.array([cc_init] + cc3)
L1 = np.cumsum([1] + l1)-1
L2 = np.cumsum([1] + l2)-1
L3 = np.cumsum([1] + l3)-1


fig, ax = plt.subplots(1,1, figsize=(7,4))
ax.plot(L1, CC1[L1], '--',color="tab:blue", label="NMA")
ax.plot(L1, CC1[L1], 'o',color="tab:blue")
ax.plot(L2, CC2[L2], '--',color="tab:green", label="NMA/HMC")
ax.plot(L2, CC2[L2], 'o',color="tab:green")
ax.plot(L3, CC3[L3], '--',color="tab:red", label="HMC")
ax.plot(L3, CC3[L3], 'o',color="tab:red")
ax.plot(CC1, color="tab:blue")
ax.plot(CC2, color="tab:green")
ax.plot(CC3, color="tab:red")
ax.set_ylabel("Cross Correlation")
ax.set_xlabel("Integrator Steps")
ax.legend()
fig.tight_layout()

fig.savefig('results/EUSIPCO/HMCNMA'+str(number)+'.png', format='png', dpi=1000)

with open('results/EUSIPCO/HMCNMA'+str(number)+'.pkl', 'wb') as f:
    dic = {
        "NMA": {
            "q_res":q_res1,
            "l":l1,
            "cc":cc1
        },
        "NMAHMC": {
            "x_res": x_res2,
            "q_res": q_res2,
            "l": l2,
            "cc": cc2
        },
        "HMC": {
            "x_res": x_res3,
            "l": l3,
            "cc": cc3
        }
    }
    pickle.dump(obj=dic, file=f)

#
#t=1060.4436702728271  iter=1129  t/iter=0.9392769628124389
#t=1322.8077182769775  iter=1542  t/iter=0.8578519706750812
# x_res1, q_res1 = src.flexible_fitting.NMA_gradient_descend(init=init, target_density=target_density,n_iter=20, dt=15,
#                                    size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=threshold, q_init = np.array([0,0,0,2000]))
#
# plot_structure([target.coords, x_res1, init.coords], ["target", "res", "init"])
#
#
# x_res2, q_res2 = src.flexible_fitting.MCNMA_gradient_descend(init=init, target_density=target_density,n_iter=200, dxt=0.001,
#                                     dqt=5, k = 100,
#                                    size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=threshold, q_init= np.array([0,0,0,700.0]))
#
# plot_structure([target.coords, x_res2, init.coords], ["target", "res", "init"])
#
#
# x_res3 = src.flexible_fitting.MC_gradient_descend(init=init, target_density=target_density,n_iter=200, dt=0.001, k=100,
#                                    size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=threshold)
#
# plot_structure([target.coords, x_res3], ["target", "res", "init"])
#
# fitted = src.molecule.Molecule(x_res3, chain_id=init.chain_id)
# test = fitted.to_density(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold)
#
# src.io.save_pdb(mol1, file = "results/P97/fitted.pdb", gen_file="data/P97/5ftm.pdb")
# src.io.save_pdb(init, file = "results/P97/init.pdb", gen_file="data/P97/5ftm.pdb")
# src.io.save_pdb(test, file = "results/P97/test.pdb", gen_file="data/P97/5ftm.pdb")
# src.io.save_density(target_density,outfilename= "results/P97/target.mrc")
#
#
# fitted = src.molecule.Molecule(coord, chain_id=init.chain_id)
# src.viewers.chimera_fit_viewer(mol1, target_density, genfile="data/P97/5ftm.pdb")
#

########################################################################
#
# from src.flexible_fitting import *
# import src.io
# from src.constants import *
#
# # import PDB
# init =src.io.read_pdb("data/P97/5ftm.pdb")
# init.center_structure()
# init.add_modes("data/P97/modes/vec.", n_modes=43)
# init.select_modes(np.array([10, 13, 28])-7)
# init = init.select_atoms(pattern='CA')
# target =src.io.read_pdb("data/P97/5ftn.pdb")
# target.center_structure()
# target = target.select_atoms(pattern='CA')
# size=128
# sampling_rate=2
# threshold=4
# gaussian_sigma=2
# target_density = target.to_density(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold)
# target_density.show()
#
# # np.sqrt(K_BOLTZMANN*Ti / (CARBON_MASS* (3 * xt.shape[0])))
# # T = K / (1 / 2 *K_BOLTZMANN )
# T =1000
#
# params ={
#     "x_init" : np.zeros(init.coords.shape),
#     "q_init" : np.zeros(init.modes.shape[1]),
#
#     "lb" : CARBON_MASS /(K_BOLTZMANN*T *3*init.n_atoms) *100,
#     "lp" : CARBON_MASS /(K_BOLTZMANN*T *3*init.n_atoms),
#     "lx" : 0,
#     "lq" : 0,
#
#     "max_iter": 1000,
#     "criterion" :True,
#
#     "dxt" : 0.0001,
#     "dqt" : 0.8,
#
#     "m_vt" : np.sqrt(K_BOLTZMANN*T /CARBON_MASS),
#     "m_wt" : 0,
# }
# fit  =FlexibleFitting(init, target_density)
# fit.HMC(mode="HMC", params=params, n_iter=5, n_warmup=None)
#
