# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from src.molecule import Molecule
import src.simulation
import src.viewers
from src.density import Volume
from src.flexible_fitting import FlexibleFitting, multiple_fitting

N=100
# test = "/home/guest/ScipionUserData/projects/synthesize/Runs/001738_FlexProtSynthesizeSubtomo/extra/"
#
# gt_modes, gt_shifts, gt_angles = src.io.read_xmd(test+"GroundTruth.xmd")
# # gt_angles = gt_angles*np.pi/180

# import PDB
init =Molecule.from_file("data/AK_tomos/AK.pdb")
init.center_structure()
fnModes = np.array(["data/AK_tomos/modes/vec."+str(i+7) for i in range(3)])
init.add_modes(fnModes)
# init.set_forcefield(psf_file="data/AK/AK.psf")
init.select_atoms(pattern='CA')
init.set_forcefield()

amp = np.random.uniform(-150,150,N)
q = np.zeros((N,3))
q[:,0] = amp
q[:,2] = amp
gt_modes = q

size=64
voxel_size=2.2
threshold= 4
gaussian_sigma=2
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)

targets=[]
# inits=[]
target_densities=[]
# for i in range(N):
#     # a = Molecule.from_molecule(init)
#     # sim = src.simulation.Simulator(init)
#     # a = sim.nma_deform(gt_modes[i])
#     # a.rotate(gt_angles[i])
#     # # a.coords += gt_shifts[i]*voxel_size
#     # inits.append(a)
#     t = Volume.from_file(file=test+str(i+1).zfill(5)+"_reconstructed.mrc",
#                                     voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
#     t.rescale(init_density)
#     targets.append(t)
for i in range(N):
    print(i)
    nma = src.simulation.nma_deform(init, q[i])
    target = src.simulation.energy_min(nma, U_lim=1063, step=0.005, verbose=False)
    targets.append(target)
    target_densities.append(Volume.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma,
                                        threshold=threshold))

# n=21
# src.viewers.chimera_fit_viewer(mol = init, target=targets[n])
########################################################################################################
#               HMC
########################################################################################################

params ={
    "lb" : 1000,
    "lp" : 1,

    "max_iter": 10,
    "criterion" :False,

    "x_dt" : 0.001,
    "x_mass" : 1,
    "x_init": np.zeros(init.coords.shape),

    "q_dt" : 0.01,
    "q_mass" : 1,
    "q_init": np.zeros(init.modes.shape[1]),

    "angles_dt": 0.00001,
    "angles_mass": 1,
    "angles_init": np.zeros(3),
    "langles": 100,

    "shift_dt": 0.00005,
    "shift_mass": 1,
    "shift_init": np.zeros(3),
    "lshift" : 100,

}
n_iter=20
n_warmup = n_iter // 2

# n=58
# src.viewers.chimera_fit_viewer(mol=init, target=target_densities[n])
#
# fit  =FlexibleFitting(init = init, target= target_densities[0], vars=["x", "q"], params=params,
#                       n_iter=n_iter, n_warmup=n_warmup, n_chain=4, verbose=2)
# fit.HMC()
# fit.show()
# fit.show_3D()
# src.viewers.chimera_molecule_viewer([fit.res["mol"], targets[n]])

models= []
for i in target_densities:
    models.append(FlexibleFitting(init=init, target=i, vars=["x", "q"], params=params,
                                                n_iter=n_iter, n_warmup=n_warmup, n_chain=4, verbose=0))


fits = multiple_fitting(models, n_chain=4, n_proc=24)

n_components=2
res_arr = np.array([i.coords.flatten() for i in targets]+ [i.res["mol"].coords.flatten() for i in fits])
res_pca = PCA(n_components=n_components)
res_pca.fit(res_arr.T)

plt.figure()
views = ['ground truth','global + local']
for i in range(len(views)):
    plt.plot(res_pca.components_[0,i*N:(i+1)*N], res_pca.components_[1,i*N:(i+1)*N], 'o',label=views[i],markeredgecolor='black')
plt.savefig("PCA.png")

# plt.figure()
# plt.plot([np.mean([j["CC"][n_warmup:] for j in i.fit]) for i in fits])
# fits2 = multiple_fitting(init=init, targets=target_densities, vars=["q"],     n_chain=2, n_iter=n_iter, n_warmup=n_warmup, params=params, n_proc=12)
# fits3 = multiple_fitting(init=init, targets=target_densities, vars=["x"],     n_chain=2, n_iter=n_iter, n_warmup=n_warmup, params=params, n_proc=12)

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

# fits= fits1 + fits2 + fits3
#
# n_components=3
# res_arr = np.array([i.coords.flatten() for i in targets]+ [i.res["mol"].coords.flatten() for i in fits])
# res_pca = PCA(n_components=n_components)
# res_pca.fit(res_arr.T)
#
#
# plt.figure()
# plt.plot(res_pca.components_[0,  0:100], res_pca.components_[1,  0:100], 'o',label='ground truth',markeredgecolor='black')
# plt.plot(res_pca.components_[0,100:200], res_pca.components_[1,100:200], 'o',label='global + local',markeredgecolor='black')
# plt.plot(res_pca.components_[0,200:300], res_pca.components_[1,200:300], 'o', label='global',markeredgecolor='black')
# plt.plot(res_pca.components_[0,300:400], res_pca.components_[1,300:400], 'o',label='local',markeredgecolor='black')
# plt.legend()
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(res_pca.components_[0,  0:100], res_pca.components_[1,  0:100], res_pca.components_[2,  0:100], s=10, label='ground truth')
# ax.scatter(res_pca.components_[0,100:200], res_pca.components_[1,100:200], res_pca.components_[2,100:200], s=10, label='global + local')
# ax.scatter(res_pca.components_[0,200:300], res_pca.components_[1,200:300], res_pca.components_[2,200:300], s=10, label='global')
# ax.scatter(res_pca.components_[0,300:400], res_pca.components_[1,300:400], res_pca.components_[2,300:400], s=10, label='local')
