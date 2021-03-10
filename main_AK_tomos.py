from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.decomposition import PCA

import src.io
import src.simulation
from src.flexible_fitting import *
from src.functions import multiple_fitting
from src.viewers import chimera_structure_viewer

N=100
test = "test_noise_no_mw"

# gt_modes, gt_shifts, gt_angles = src.io.read_xmd("data/AK_tomos/"+test+"/GroundTruth.xmd")
# gt_angles = gt_angles*np.pi/180

# import PDB
init =src.io.read_pdb("data/AK_tomos/AK.pdb")
# init.center_structure()
init.add_modes("data/AK_tomos/modes/vec.", n_modes=3)
# init.set_forcefield(psf_file="data/AK/AK.psf")
init.select_atoms(pattern='CA')
init.set_forcefield()
#
amp = np.random.uniform(-150,150,N)
q = np.zeros((N,3))
q[:,0] = amp
q[:,2] = amp
gt_modes = q

size=64
voxel_size=1.5
threshold= 4
gaussian_sigma=2
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)

targets=[]
inits=[]
target_densities=[]
# for i in range(N):
    # # a = Molecule.from_molecule(init)
    # sim = src.simulation.Simulator(init)
    # a = sim.nma_deform(gt_modes[i])
    # # a.rotate(gt_angles[i])
    # # # a.coords += gt_shifts[i]*voxel_size
    # inits.append(a)
    #
    # targets.append(Volume.from_file(file="data/AK_tomos/"+test+"/"+str(i+1).zfill(5)+"_reconstructed.mrc",
    #                                 voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold))
for i in range(N):
    print(i)
    sim = src.simulation.Simulator(init)
    nma = sim.nma_deform(q[i])
    target = nma.energy_min(U_lim=1063, step=0.005, verbose=False)
    targets.append(target)
    target_densities.append(Volume.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma,
                                        threshold=threshold))

# n=21
# chimera_fit_viewer(mol = init, target=targets[n])
########################################################################################################
#               HMC
########################################################################################################

params ={
    "x_init" : np.zeros(init.coords.shape),
    "q_init" : np.zeros(init.modes.shape[1]),

    "lb" : 50, #np.mean(init.prm.mass)*ATOMIC_MASS_UNIT /(K_BOLTZMANN*T *3*init.n_atoms) *1e6,
    "lp" : 1, #np.mean(init.prm.mass)*ATOMIC_MASS_UNIT /(K_BOLTZMANN*T *3*init.n_atoms)*1e6,

    "max_iter": 20,
    "criterion" :True,

    "x_dt" : 0.005,
    "q_dt" : 0.05,

    "x_mass" :  1,#np.sqrt(K_BOLTZMANN*T /(np.mean(init.prm.mass)*ATOMIC_MASS_UNIT)),
    # np.array([np.sqrt(K_BOLTZMANN*T /(init.prm.mass*ATOMIC_MASS_UNIT)),
    #           np.sqrt(K_BOLTZMANN*T /(init.prm.mass*ATOMIC_MASS_UNIT)),
    #           np.sqrt(K_BOLTZMANN*T /(init.prm.mass*ATOMIC_MASS_UNIT))]),
    "q_mass" : 1,
}
n_iter=20
n_warmup = n_iter // 2

n=0
fit  =FlexibleFitting(init = init, target= target_densities[n], mode="HMCNMA", params=params,
                      n_iter=n_iter, n_warmup=n_warmup, n_chain=4, verbose=2)
fit.HMC()
fit.show()
fit.show_3D()
chimera_structure_viewer([fit.res["mol"], init])


fits = multiple_fitting(init=init, targets=target_densities, mode="HMC", n_chain=4, n_iter=n_iter, n_warmup=n_warmup, params=params, n_proc=12)
fits2 = multiple_fitting(init=init, targets=target_densities, mode="NMA", n_chain=4, n_iter=n_iter, n_warmup=n_warmup, params=params, n_proc=12)
fits3 = multiple_fitting(init=init, targets=target_densities, mode="HMCNMA", n_chain=4, n_iter=n_iter, n_warmup=n_warmup, params=params, n_proc=12)

# plt.figure()
# plt.plot(np.arange(-200,200,1),np.arange(-200,200,1))
# q_res=[]
# for i in range(N):
#     f = fits[i]
#     q_res .append(f.res["q"])
#     plt.plot(f.res["q"][0], f.res["q"][2], "o", color="b")
#     plt.plot(gt_modes[i][0], gt_modes[i][2], "o", color="orange")
# plt.xlabel("Mode 7")
# plt.ylabel("Mode 9")


n_components=3
target_arr = np.array([i.coords.flatten() for i in targets])
target_pca = PCA(n_components=n_components)
target_pca.fit(target_arr.T)
res_arr = np.array([i.res["mol"].coords.flatten() for i in fits])
res_pca = PCA(n_components=n_components)
res_pca.fit(res_arr.T)
res2_arr = np.array([i.res["mol"].coords.flatten() for i in fits2])
res2_pca = PCA(n_components=n_components)
res2_pca.fit(res2_arr.T)
res3_arr = np.array([i.res["mol"].coords.flatten() for i in fits3])
res3_pca = PCA(n_components=n_components)
res3_pca.fit(res3_arr.T)

plt.figure()
plt.plot(target_pca.components_[0], target_pca.components_[1],  'o',label='ground truth',markeredgecolor='black')
plt.plot(res_pca.components_[0], res_pca.components_[1], 'o',label='global',markeredgecolor='black')
plt.plot(res2_pca.components_[0], res2_pca.components_[1], 'o',label='local',markeredgecolor='black')
plt.plot(res3_pca.components_[0], res3_pca.components_[1], 'o', label='global + local',markeredgecolor='black')
plt.legend()
plt.xlabel("Mode 7")
plt.ylabel("Mode 9")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(target_pca.components_[0], target_pca.components_[1],target_pca.components_[2], s=10,label='ground truth')
ax.scatter(res_pca.components_[0] , res_pca.components_[1] ,res_pca.components_[2]       , s=10,label='global')
ax.scatter(res2_pca.components_[0], res2_pca.components_[1],res2_pca.components_[2]      , s=10,label='local')
ax.scatter(res3_pca.components_[0], res3_pca.components_[1],res3_pca.components_[2]      , s=10,label='global + local')