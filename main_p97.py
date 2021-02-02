import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting
from src.viewers import structures_viewer, chimera_structure_viewer
from src.flexible_fitting import *
import src.molecule
import src.io

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

# mol, q_res = HNMA(init=init, q_init = None, target_density=target_density, n_iter = 20, k=1, dt=5)

# init.select_modes(np.array([0]))

mol1, x_res1, q_res1, U1, L1, lx, lq= HMCNMA(init=init, q_init = None, target_density=target_density, n_iter = 20, n_warmup=10, max_iter=300, k=100, dxt=0.02, dqt=0.2, m_test=100)

# #################################
# # TEST
# fig, ax = plt.subplots(1,1)
#
# x_nma = np.zeros(len(lq)-1)
# x_mc = np.zeros(len(lq)-1)
# for i in range(len(lq)-1):
#     a = np.dot(lq[i], init.modes)
#     b = np.dot(lq[i+1], init.modes)
#     x_nma[i] = np.linalg.norm(a-b)
#     x_mc[i] = np.linalg.norm(lx[i]-lx[i+1])
# ax.plot(np.cumsum(x_nma))
# ax.plot(np.cumsum(x_mc))


###################################
# mol, x_res, q_res = HMCNMA2(init=init, q_init = None, target_density=target_density, n_iter = 10, n_warmup=5, max_iter=100,
#                     lambda1=100, lambda2=1, lambda3=0,lambda4=0, dxt=0.02, dqt=0.5)

mol2, x_res2, U2, L2 = HMC(init=init, target_density=target_density, n_iter = 20, n_warmup=10, k=100, dt=0.02, max_iter=300)

L1_arr = np.cumsum(np.array([1] + L1)-1)
L2_arr = np.cumsum(np.array([1] + L2)-1)
U1_arr = -np.array(U1)
U2_arr = -np.array(U2)

fig, ax = plt.subplots(1,1, figsize=(7,4))
ax.plot(L1_arr, U1_arr[L1_arr], '--',color="tab:blue", label="NMA/HMC")
ax.plot(L1_arr, U1_arr[L1_arr], 'o',color="tab:blue")
ax.plot(L2_arr, U2_arr[L2_arr], '--',color="tab:green", label="HMC")
ax.plot(L2_arr, U2_arr[L2_arr], 'o',color="tab:green")
ax.plot(U1_arr, color="tab:blue")
ax.plot(U2_arr, color="tab:green")
ax.set_ylabel("Log Posterior")
ax.set_xlabel("Leap-frog Steps")
ax.legend()
fig.tight_layout()
# fig.savefig('results/mc_vs_mcnma.png', format='png', dpi=1000)



cc1 = np.zeros(x_res1.shape[0]+1)
cc2 = np.zeros(x_res1.shape[0]+1)
cc1[0] = cross_correlation(init.to_density(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold).data, target_density.data)
cc2[0] = cross_correlation(init.to_density(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold).data, target_density.data)
for i in range(x_res1.shape[0]):
    # Volume update
    coordt=init.coords + x_res1[i] + np.dot(q_res1[1], init.modes)
    psim = volume_from_pdb_fast3(coord=coordt, size=target_density.size, sampling_rate=target_density.sampling_rate,
                                 sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
    cc1[i+1] = cross_correlation(psim, target_density.data)

    coordt = x_res2[i]
    psim = volume_from_pdb_fast3(coord=coordt, size=target_density.size, sampling_rate=target_density.sampling_rate,
                                 sigma=target_density.gaussian_sigma, threshold=target_density.threshold)
    cc2[i+1] = cross_correlation(psim, target_density.data)

fig, ax = plt.subplots(1,1, figsize=(7,4))
ax.plot(L1_arr, cc1, '--',color="tab:blue", label="NMA/HMC")
ax.plot(L1_arr, cc1, 'o',color="tab:blue")
ax.plot(L2_arr, cc2, '--',color="tab:green", label="HMC")
ax.plot(L2_arr, cc2, 'o',color="tab:green")
# ax.plot(U1_arr, color="tab:blue")
# ax.plot(U2_arr, color="tab:green")
ax.set_ylabel("Cross Correlation")
ax.set_xlabel("Integrator Steps")
ax.legend()
fig.tight_layout()


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

x_res3 = src.flexible_fitting.MC_gradient_descend(init=init, target_density=target_density,n_iter=200, dt=0.001, k=100,
                                   size=size, sigma=gaussian_sigma, sampling_rate=sampling_rate, threshold=threshold)

plot_structure([target.coords, x_res3], ["target", "res", "init"])

fitted = src.molecule.Molecule(x_res3, chain_id=init.chain_id)
test = fitted.to_density(size=size, sampling_rate=sampling_rate, gaussian_sigma=gaussian_sigma, threshold=threshold)

src.io.save_pdb(mol1, file = "results/P97/fitted.pdb", gen_file="data/P97/5ftm.pdb")
src.io.save_pdb(init, file = "results/P97/init.pdb", gen_file="data/P97/5ftm.pdb")
src.io.save_pdb(test, file = "results/P97/test.pdb", gen_file="data/P97/5ftm.pdb")
src.io.save_density(target_density,outfilename= "results/P97/target.mrc")


fitted = src.molecule.Molecule(coord, chain_id=init.chain_id)
src.viewers.chimera_fit_viewer(mol1, target_density, genfile="data/P97/5ftm.pdb")

