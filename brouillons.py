# import time
#
# import numpy as np
#
# import src.io
# from src.forcefield import *
# from src.functions import *
#
# init =src.io.read_pdb("data/AK_tomos/AK.pdb")
# init.add_modes("data/AK_tomos/modes/vec.", n_modes=4)
# init.select_atoms(pattern='CA')
# init.set_forcefield()
#
#
# x = np.zeros(3)
# q = np.zeros(init.modes.shape[1])
# angles = np.random.uniform(-0.5,0.5,3)
#
#
# R = generate_euler_matrix(angles = angles)
#
# t = time.time()
# dangles1 =get_euler_autograd(coord = init.coords[0], A = init.modes[0], x=x, q=q, angles=angles)
# print(time.time() - t)
#
# t = time.time()
# dangles2 =get_euler_grad(angles=angles, coord = init.coords[0])
# print(time.time() - t)
#
# print(dangles1[2] - np.sum(dangles2, axis=1))
#
# d = {"x": np.array([0.0,0.0,0.0]), "angles": np.array([0.0,0.1,0.2])}
# a = get_autograd(init, d)
#
# ########################################################################################################################
#
# import src.simulation
# from src.viewers import chimera_structure_viewer
# from src.flexible_fitting import *
# from src.molecule import Molecule
# import copy
#
#
# init =Molecule.from_file("data/AK/AK_PSF.pdb")
# init.center_structure()
# init.add_modes("data/AK/modes_psf/vec.", n_modes=4)
# init.select_atoms(pattern='CA')
# init.set_forcefield()
#
#  # q = [300,-100,0,0]
# q = [0,0,0,0]
# # angles= [-0.1,0.2, 0.5]
# target = src.simulation.nma_deform(init, q)
# # target.rotate(angles)
# shift = [8.3,5.2,7.0]
# target.coords += shift
#
# size=64
# voxel_size=2.2
# threshold= 4
# gaussian_sigma=2
# target_density = Image.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
# init_density = Image.from_coords(coord=init.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
# # init_density.show()
# # target_density.show()
#
# params ={
#     "lb" : 100,
#     "lp" : 1,
#     "max_iter": 10,
#     "criterion" :False,
#
#     "q_init" : np.zeros(4),
#     "q_dt" : 0.1,
#     "q_mass" : 1,
#
#     "angles_init" : np.zeros(3),
#     "angles_dt" : 0.0001,
#     "angles_mass" : 1,
#     "langles" :0,
#
#     "shift_init": np.zeros(3),
#     "shift_dt": 0.001,
#     "shift_mass": 1,
#     "lshift": 0,
# }
# n_iter=20
# n_warmup = n_iter // 2
#
# fit  =FlexibleFitting(init = init, target= target_density, vars=["shift"], params=params,
#                       n_iter=n_iter, n_warmup=n_warmup, n_chain=1, verbose=2)
#
# fit.HMC()
# fit.show()
# fit.show_3D()
# chimera_structure_viewer([fit.res["mol"], target, init])
#
#
# test = copy.deepcopy(init)
# test.rotate(fit.res["angles"])
# chimera_structure_viewer([test, target])
#
#
#
#
# ############################################################
# ###  AUTOGRAD
# ############################################################
#
#
#
#
#
# import autograd.numpy as npg
# from autograd import elementwise_grad
#
# def get_energy_autograd(mol, params):
#     coord = npg.array(mol.coords)
#     if "x" in params:
#         coord+= params["x"]
#     if "q" in params:
#         coord+= npg.dot(params["q"], mol.modes)
#     if "angles" in params:
#         coord = npg.dot(src.functions.generate_euler_matrix(params["angles"]), coord.T).T
#
#     U_bonds = get_energy_bonds(coord, mol.bonds, mol.prm)
#     U_angles = get_energy_angles(coord, mol.angles, mol.prm)
#     U_dihedrals = get_energy_dihedrals(coord, mol.dihedrals, mol.prm)
#
#     return U_bonds + U_angles + U_dihedrals
#
# def get_autograd(mol, params):
#     grad = elementwise_grad(get_energy_autograd, 1)
#     return grad(mol, params)
# d = {"x":np.array([0.0,0.0,0.0]),
#      "q": np.array([0.0,0.0,0.0,0.0]),
#      "angles" : np.array([0.0,0.0,0.0])}
# get_energy_autograd(init,  d)
# a = get_autograd(init,  d)
#
# from src.forcefield import get_gradient_auto
# get_gradient_auto(init, {"x": init.coords, "angles": [0.0,.0,.0]})
#
#
# ############################################################
# ###  FIND CHAIN IN PSEUDOATOMS
# ############################################################
#
# from src.molecule import Molecule
# from src.viewers import chimera_molecule_viewer
# import numpy as np
# import matplotlib.pyplot as plt
# import copy
#
# mol =Molecule.from_file(file="/home/guest/ScipionUserData/projects/BayesianFlexibleFitting/Runs/001192_FlexProtConvertToPseudoAtoms/pseudoatoms.pdb")
#
# dist = np.zeros((mol.n_atoms, mol.n_atoms))
# for i in range(mol.n_atoms):
#     for j in range(mol.n_atoms):
#         dist[i,j] = np.linalg.norm(mol.coords[i]-mol.coords[j])
#         if i==j :
#             dist[i,j]=100.0
# plt.pcolormesh(dist)
# np.where(dist==dist.min())
#
# mol.set_forcefield()
# U =mol.get_energy()
#
# n=0
# l=[]
# while(n<1000):
#     for i in range(mol.n_atoms-1):
#         molc = copy.deepcopy(mol)
#         tmp1 = copy.deepcopy(molc.coords[i])
#         tmp2 = copy.deepcopy(molc.coords[i+1])
#         molc.coords[i] = tmp2
#         molc.coords[i+1] = tmp1
#         molc.set_forcefield()
#         Uc = molc.get_energy()
#         if Uc < U:
#             print("yes"+str(n)+" ; "+str(np.mean(l)))
#             mol = molc
#             U=Uc
#             l.append(1)
#
#         else :
#             l.append(0)
#         if len(l)>20:
#             l=l[1:]
#     n+=1
#
#
#
# ############################################################
# ###  P97
# ############################################################
#
# from src.molecule import Molecule
# import numpy as np
# import copy
# from src.viewers import chimera_molecule_viewer
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
#
# init = Molecule.from_file("data/P97/5ftm.pdb")
# init.center_structure()
# fnModes = np.array(["data/P97/modes_atoms/vec."+str(i+7) for i in range(4)])
# init.add_modes(fnModes)
#
# N=1000
# mols = []
# q = np.random.randint(0,2,(N,6))
# for n in range(N):
#     mol = copy.deepcopy(init)
#     for i in range(mol.n_chain):
#         mol.coords[mol.chain_id[i]:mol.chain_id[i+1]] += np.dot(np.array([0,0,q[n,i]*-1500,0]), mol.modes[mol.chain_id[i]:mol.chain_id[i+1]])
#     mols.append(mol)
#
# # chimera_molecule_viewer([mols[1]])
#
#
# n_components=mol.n_chain
# res_arr = np.array([i.coords.flatten() for i in mols])
# res_pca = PCA(n_components=n_components)
# res_pca.fit(res_arr.T)
#
# i=0
# plt.figure()
# plt.plot(res_pca.components_[0+i], res_pca.components_[1+i], 'o',label='ground truth',markeredgecolor='black')
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(res_pca.components_[0], res_pca.components_[1], res_pca.components_[2], s=10, label='ground truth')
#
#
# ############################################################
# ###  PRODY
# ############################################################
#
# from src.molecule import Molecule
# import numpy as np
# import copy
# from src.viewers import chimera_molecule_viewer,chimera_fit_viewer
# from src.constants import *
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from src.functions import compute_pca
# from src.density import Volume
# from src.flexible_fitting import FlexibleFitting, multiple_fitting
#
# init = Molecule.from_file("data/AK/AK_prody/AK.pdb")
# init.center_structure()
# fnModes = np.array(["/home/guest/ScipionUserData/projects/nma_calculator/Runs/000759_FlexProtNMA/modes/vec."+str(i+7) for i in range(6)])
# init.add_modes(fnModes)
# init.select_atoms(pattern="CA")
# init.set_forcefield()
# init_density = Volume.from_coords(coord=init.coords, voxel_size=1.5, size=64, threshold=5, sigma=2)
#
# N=41
# mols=[]
# targets = []
# for i in range(N):
#     mol = Molecule.from_file("/home/guest/Downloads/comd_tutorial_files/frame_reduced_"+str(i)+".pdb")
#     mol.select_atoms()
#     mols.append(mol)
#     target = Volume.from_coords(coord=mol.coords, voxel_size=1.5, size=64, threshold=5, sigma=2)
#     targets.append(target)
#
# params ={
#     "biasing_factor" : 100,
#     "potential_factor" : 1,
#
#     "local_dt" : 0.01,
#     "global_dt" : 0.1,
#     "shift_dt" : 0.0001,
#
#     "n_iter":20,
#     "n_warmup":10,
#     "n_step": 20,
#     "criterion": False,
#
#     "local_mass" :  1
# }
#
# n_chain = 4
# n_proc = 12
# models_global=[]
# models_local =[]
# models_glocal=[]
# for i in targets:
#     models_global.append(FlexibleFitting(init=init, target=i, vars=["global", "shift"], params=params,n_chain=n_chain, verbose=0))
#     models_local .append(FlexibleFitting(init=init, target=i, vars=["local", "shift"], params=params,n_chain=n_chain, verbose=0))
#     models_glocal.append(FlexibleFitting(init=init, target=i, vars=["local", "global", "shift"], params=params,n_chain=n_chain, verbose=0))
#
# # n_test = 40
# # models[n_test].HMC_chain()
# # models[n_test].show()
# # models[n_test].show_3D()
# fits_global = multiple_fitting(models_global, n_chain=n_chain, n_proc=n_proc)
# fits_local = multiple_fitting(models_local, n_chain=n_chain, n_proc=n_proc)
# fits_glocal = multiple_fitting(models_glocal, n_chain=n_chain, n_proc=n_proc)
#
# n=39
# chimera_fit_viewer(fits_glocal[n].res["mol"], targets[n])
# # chimera_molecule_viewer([mols[0]]), mols[10], mols[20], mols[30], mols[40]])
#
# pca_data = [i.coords.flatten() for i in mols] + [i.res["mol"].coords.flatten() for i in fits_global]+ \
#                                                 [i.res["mol"].coords.flatten() for i in fits_local]+ \
#                                                 [i.res["mol"].coords.flatten() for i in fits_glocal]
# length= [N,N,N,N]
# labels=["Ground Truth", "Global", "Local", "Global + Local"]
# compute_pca(data=pca_data, length=length, labels= labels, n_components=2)
#
#
#
# ############################################################
# ###  TEMP, kINETIC ETC
# ############################################################
# import matplotlib.pyplot as plt
# import numpy as np
#
# from src.molecule import Molecule
# from src.density import Volume
# from src.constants import *
# import src.forcefield
#
# init = Molecule.from_file("data/AK/AK_prody/output.pdb")
# init.center_structure()
# fnModes = np.array(["data/AK/AK_prody/modes_psf/vec."+str(i+7) for i in range(6)])
# init.add_modes(fnModes)
# # init.select_atoms(pattern="CA")
# init.set_forcefield("data/AK/AK_prody/output.psf")
# voxel_size=2.0
# init_density = Volume.from_coords(coord=init.coords, voxel_size=voxel_size, size=64, threshold=5, sigma=2)
#
#
# T=300
# m = (init.prm.mass*ATOMIC_MASS_UNIT)   # kg
# sigma = (np.ones((3,init.n_atoms)) * np.sqrt((K_BOLTZMANN *T)/ (init.prm.mass*ATOMIC_MASS_UNIT))*1e10).T
# v = np.random.normal(0,sigma, init.coords.shape)          # A * s-1
# K = src.forcefield.get_kinetic(v, init.prm.mass)   # kg * A2 * s-2
# K_J = K*1e-20
# K_kcalmol =AVOGADRO_CONST* K_J /KCAL_TO_JOULE
#
# T_hat =  src.forcefield.get_instant_temp(K, init.n_atoms)   # K
# print(T_hat)
#
# U_kcalmol= init.get_energy() # kcal * mol-1
# U_jmol = (KCAL_TO_JOULE * U_kcalmol) # J * mol-1
# U_j = U_jmol/AVOGADRO_CONST # J
#
# F_kcalmolA = src.forcefield.get_autograd(params={"local":np.zeros(init.coords.shape)}, mol=init)["local"] # kcal * mol-1 * A-1
# F_JA = (KCAL_TO_JOULE * F_kcalmolA) / AVOGADRO_CONST                                             # kg * m2 * s-2 - A-1
# F = F_JA * (1e10)**2  #kg * A * s-2
# a = (F.T*(1/m)).T      #A * s-2
#
# H=U+K
#
#
# f1 = K_J/U_j
# f2 = K_kcalmol/U_kcalmol
#
#
#
#
# import os
# import matplotlib.pyplot as plt
# import numpy as np
#
# from src.molecule import Molecule
# from src.density import Volume
# from src.constants import *
# from src.simulation import nma_deform
# import src.forcefield
# from src.viewers import chimera_molecule_viewer
#
# pdb = "data/AK/AK_prody/output.pdb"
# modes = "/home/guest/ScipionUserData/projects/nma_calculator/Runs/000839_FlexProtNMA/modes2.xmd"
# new_pdb = "mol.pdb"
# q= np.zeros(14)
# q[0] = 300
# q[2] = -100
#
#
# mol1 = Molecule.from_file(pdb)
# fnModes = np.array(["/home/guest/ScipionUserData/projects/nma_calculator/Runs/000839_FlexProtNMA/modes/vec."+str(i+7) for i in range(len(q))])
# mol1.add_modes(fnModes)
# mol1 = nma_deform(mol1, q)
# mol1.show()
# mol1.save_pdb(file= "data/AK/AK_prody/AK1.pdb")
#
# cmd = "/home/guest/xmipp-bundle/xmipp/build/bin/xmipp_pdb_nma_deform"
# cmd +=" --pdb "+ pdb
# cmd +=" --nma "+ modes
# cmd +=" -o "+ new_pdb
# cmd +=" --deformations "+ ' '.join(str(i) for i in q)
# print(cmd)
#
# mol2 = Molecule.from_file(new_pdb)
# chimera_molecule_viewer([mol1, mol2])
#
#
#
# import mkl
# mkl.set_num_threads(1)
# from src.molecule import Molecule
# from src.simulation import nma_deform
# from src.flexible_fitting import *
# from src.viewers import molecule_viewer, chimera_molecule_viewer, chimera_fit_viewer
# from src.density import Volume
# from src.constants import *
#
# ########################################################################################################
# #               IMPORT FILES
# ########################################################################################################
#
# # import PDB
# init = Molecule.from_file("data/1AKE/1ake_chainA_psf.pdb")
# init.center_structure()
# fnModes = np.array(["data/1AKE/modes/vec."+str(i+7) for i in range(10)])
# init.add_modes(fnModes)
#
# target =  Molecule.from_file("data/4AKE/4ake_fitted.pdb")
# target.center_structure()
#
# init.set_forcefield(psf_file="data/1AKE/1ake_chainA.psf")
# # init.select_atoms()
# # init.set_forcefield()
# # target.select_atoms()
#
#
# size=64
# sampling_rate=1.5
# threshold= 4
# gaussian_sigma=2
# target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold)
# init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold)
# target_density.show()
#
# chimera_molecule_viewer([target, init])
# # chimera_fit_viewer(init, target_density)
#
# ########################################################################################################
# #               HMC
# ########################################################################################################
# params ={
#     "initial_biasing_factor" : 100,
#     "n_step": 20,
#
#     "local_dt" : 1e-15,
#     "temperature" : 300,
#
#     "global_dt" : 0.1,
#     "rotation_dt" : 0.00001,
#     "shift_dt" : 0.00001,
#     "n_iter":40,
#     "n_warmup":30,
# }
#
#
# fit  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL, FIT_VAR_ROTATION, FIT_VAR_SHIFT], params=params, n_chain=4, verbose=2)
#
# fit.HMC()
# fit.show()
# fit.show_3D()
# chimera_molecule_viewer([fit.res["mol"], target])
#
# data= []
# length = []
# for j in [[i.flatten() for i in n["coord_t"]] for n in fit.fit]:
#     data += j
#     length += [len(j)]
# src.functions.compute_pca(data=data, length=length, n_components=2)
#
#
#
#
#
#
#
# import numpy as np
#
# from src.molecule import Molecule
# from src.simulation import nma_deform
# from src.flexible_fitting import *
# from src.viewers import molecule_viewer, chimera_molecule_viewer, chimera_fit_viewer
# from src.density import Volume
# from src.constants import *
#
# ########################################################################################################
# #               IMPORT FILES
# ########################################################################################################
#
# # import PDB
# init = Molecule.from_file("data/ATPase/prody/1iwo_fitted.pdb")
# init.center_structure()
# target =  Molecule.from_file("data/ATPase/prody/1su4.pdb")
# target.center_structure()
#
# size=128
# sampling_rate=1.5
# threshold= 4
# gaussian_sigma=2
# target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold)
# init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold)
#
# def get_grad_RMSD3(coord, psim, pexp, size, sampling_rate, sigma, threshold):
#     vox, n_vox = src.functions.select_voxels(coord, size, sampling_rate, threshold)
#     n_atoms = coord.shape[0]
#     pdiff = psim - pexp
#
#
#     dx = np.zeros(coord.shape)
#     for i in range(n_atoms):
#         mu = (np.mgrid[vox[i, 0]:vox[i, 0] + n_vox,
#               vox[i, 1]:vox[i, 1] + n_vox,
#               vox[i, 2]:vox[i, 2] + n_vox] - size / 2) * sampling_rate
#         x = np.repeat(coord[i], n_vox ** 3).reshape(3, n_vox, n_vox, n_vox)
#         tmp = 2* pdiff[vox[i,0]:vox[i,0]+n_vox,
#             vox[i,1]:vox[i,1]+n_vox,
#             vox[i,2]:vox[i,2]+n_vox]*np.exp(-np.square(np.linalg.norm(x-mu, axis=0))/(2*(sigma ** 2)))
#         dpsim =-(1/(sigma**2)) * (x-mu) * np.array([tmp,tmp,tmp])
#         dx[i] = np.sum(dpsim, axis=(1,2,3))
#
#     return dx
#
# a = target_density.get_gradient_RMSD(mol=init, psim=init_density.data, params={"local":np.ones((init.n_atoms, 3))})["local"]
# b = get_grad_RMSD3(coord= init.coords+np.ones((init.n_atoms, 3)), psim=init_density.data, pexp=target_density.data, size=target_density.size, sampling_rate=target_density.voxel_size,
#                    sigma=target_density.sigma, threshold=target_density.threshold)
#
#
#
#
# import numpy as np
#
# from src.molecule import Molecule
# from src.simulation import nma_deform
# from src.flexible_fitting import *
# from src.viewers import molecule_viewer, chimera_molecule_viewer, chimera_fit_viewer
# from src.density import Volume
# from src.constants import *
# # import PDB
# init = Molecule("data/P97/5ftm_psf.pdb")
#
# residx = []
# reslist = list(set(init.resnum))
# reslist.sort()
# for i in reslist:
#     residx.append(np.where(init.resnum == i)[0])
#
# def get_dist_res(mol, residx):
#     n_res = len(residx)
#     rescoords = np.zeros((n_res,3))
#     for i in range(n_res):
#         rescoords[i] = np.mean(mol.coords[residx[i]], axis=0)
#     return rescoords
#     #
#     # dist = np.linalg.norm(np.repeat(rescoords,n_res, axis=0).reshape(n_res, n_res, 3) - rescoords,
#     #                       axis=2)
#     # return dist
#
# a = get_dist_res(init, residx)
# mol = Molecule(coords=a)
# mol.show()
#
#
#
#
#
#
# from src.flexible_fitting import FlexibleFitting
#
# fitq =  FlexibleFitting.load("results/P97/p97_allatoms_exp50_fitq_output.pkl")
# fitx =  FlexibleFitting.load("results/P97/p97_allatoms_exp50_fitx_output.pkl")
# fitxq = FlexibleFitting.load("results/P97/p97_allatoms_exp50_fitxq_output.pkl")
#
# fits = [fitx, fitq, fitxq]
#
# import matplotlib.pyplot as plt
# from src.functions import cross_correlation
# from matplotlib.ticker import MaxNLocator
# # cc_init= cross_correlation(init_density.data, target_density.data)
# fig, ax = plt.subplots(1,1, figsize=(5,2))
# ax.plot(np.mean([i["CC"] for i in fits[0].fit], axis=0), '-', color="tab:red", label="local")
# ax.plot(np.mean([i["CC"] for i in fits[1].fit], axis=0), '-', color="tab:blue", label="global")
# ax.plot(np.mean([i["CC"] for i in fits[2].fit], axis=0), '-', color="tab:green", label="local + global")
# ax.set_ylabel("Correlation Coefficient")
# ax.set_xlabel("HMC iteration")
# ax.legend(loc="lower right", fontsize=9)
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# fig.tight_layout()
# fig.savefig("/home/guest/Documents/Screeshots/P97/synth50_CC.png")
#
# import numpy as np
# from src.molecule import Molecule
# import time
# #
# # mol = Molecule("data/AK/AK_PSF.pdb")
# # mol.set_forcefield(psf_file="data/AK/AK.psf", prm_file="data/toppar/par_all36_prot.prm")
# # mol.get_energy(verbose=True)
# mol = Molecule("data/P97/5ftm_psf.pdb")
# mol.set_forcefield(psf_file="data/P97/5ftm.psf", prm_file="data/toppar/par_all36_prot.prm")
# mol.get_energy(verbose=True)
#
# idx_reschain=[]
# for i in set(mol.chainName):
#     idx_chain = np.where(mol.chainName == i)[0]
#     for j in set(mol.resNum):
#         idx_res =np.where(mol.resNum == j)[0]
#         idx_reschain.append(np.intersect1d(idx_chain, idx_res))
#
# coords_reschain = np.array([np.mean(mol.coords[i], axis=0) for i in idx_reschain])
#
#
# t = time.time()
#
# vdw_a= []
# vdw_b= []
# for i in range(coords_reschain.shape[0]):
#     dist_reschain = np.linalg.norm(coords_reschain[i+1:] - coords_reschain[i], axis=1)
#     vdw_reschain = np.where(dist_reschain < cutoff)[0]+i
#     print(i)
#
#     idx = []
#     for j in vdw_reschain:
#         idx += list(idx_reschain[j])
#     idx=np.array(idx)
#     for j in range(len(idx)):
#         dist = np.linalg.norm(mol.coords[idx] - mol.coords[idx[j]], axis=1)
#         dist[j] =100
#
#         idx_tmp =idx[np.where(dist < cutoff)[0]]
#         vdw_a += list(np.full(shape=len(idx_tmp), fill_value=idx[j], dtype=np.int))
#         vdw_b += list(idx_tmp)
#
# vdw = np.array([vdw_a,vdw_b])
# t1 = time.time()-t
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

import numpy as np
from src.molecule import Molecule
import time
import autograd.numpy as npg
from autograd import elementwise_grad
from src.forcefield import *
from src.constants import *
import matplotlib.pyplot as plt

mol = Molecule("data/AK/AK_PSF.pdb")
mol.set_forcefield(psf_file="data/AK/AK.psf", prm_file="data/toppar/par_all36_prot.prm")
mol.get_energy(verbose=True, cutoff=25.0)
mol.forcefield.epsilon[-1] *  KCAL_TO_JOULE / (AVOGADRO_CONST*K_BOLTZMANN)

def get_pairlist(coord,cutoff=10.0):
    print("Building pairlist ...")
    pairlist = []
    for i in range(coord.shape[0]):
        dist = np.linalg.norm(coord[i + 1:] - coord[i], axis=1)
        idx = np.where(dist < cutoff)[0] + i + 1
        for j in idx:
            pairlist.append([i, j])
    print("Converting to array ..." + str(sys.getsizeof(pairlist) / (8 * 1024)) + " kB")
    pl_arr= np.array(pairlist)
    print("Done : " + str(sys.getsizeof(pl_arr) / (8 * 1024)) + " kB")
    return pl_arr

pl = get_pairlist(mol.coords, cutoff=10.0)

invdist = get_invdist(mol.coords, pairlist=pl)
Rminij = mol.forcefield.Rmin[pl[:,0]] + mol.forcefield.Rmin[pl[:,1]]
Epsij = npg.sqrt(mol.forcefield.epsilon[pl[:,0]] * mol.forcefield.epsilon[pl[:,1]])
invdist6 = (Rminij*invdist) ** 6
invdist12 = invdist6 ** 2
U_vdw = Epsij * (invdist12 - 2*invdist6)

pl2 = np.concatenate((mol.forcefield.bonds, mol.forcefield.angles[:,[0,2]]))
invdist2 = get_invdist(mol.coords, pairlist=pl2)
Rminij2 = mol.forcefield.Rmin[pl2[:,0]] + mol.forcefield.Rmin[pl2[:,1]]
Epsij2 = npg.sqrt(mol.forcefield.epsilon[pl2[:,0]] * mol.forcefield.epsilon[pl2[:,1]])
invdist62 = (Rminij2*invdist2) ** 6
invdist122 = invdist62 ** 2
U_vdw2 =  Epsij2 * (invdist122 - 2*invdist62)

plt.plot(1/(invdist),U_vdw/Epsij, "x")
plt.plot(1/(invdist2),U_vdw2/Epsij2, "x", color="g")
plt.axhline(0, color="r")
plt.xlim(0.9,10.0)
plt.ylim(-1.5,1.5)

print(U_vdw.sum() - U_vdw2.sum())

mol = Molecule("data/P97/5ftm_psf.pdb")
mol = Molecule("results/P97/p97_allatoms_exp_fitxq_chain0.pdb")
mol.set_forcefield(psf_file="data/P97/5ftm.psf", prm_file="data/toppar/par_all36_prot.prm")
mol.get_energy(verbose=True)

pairlist = get_pairlist(mol.coords,cutoff = 10.0)
F_auto = get_autograd(params={"local":np.zeros(mol.coords.shape)}, mol=mol, potentials=["vdw"], pairlist=pairlist)["local"]

invdist = get_invdist(mol.coords, pairlist)
dist = 1/invdist
Rminij = mol.forcefield.Rmin[pairlist[:, 0]] + mol.forcefield.Rmin[pairlist[:, 1]]
Epsij = npg.sqrt(mol.forcefield.epsilon[pairlist[:, 0]] * mol.forcefield.epsilon[pairlist[:, 1]])

invdist6 = (Rminij * invdist) ** 6
invdist12 = invdist6 ** 2

vdw = -12 * Epsij * invdist * (invdist12 - invdist6) * invdist * 0.00004

F = np.zeros(mol.coords.shape)
for i in range(len(pairlist)):
    f = vdw[i] *mol.coords[pairlist[i, 0]]
    F[pairlist[i, 0]] += f
    F[pairlist[i, 1]] -= f

plt.figure()
plt.plot(dist, vdw)
plt.axhline(0)

from src.functions import compute_pca
from src.molecule import Molecule
from src.simulation import nma_deform

mol = Molecule("data/AK/AK_PSF.pdb")
target = Molecule("data/AK/AK_PSF_deformed.pdb")

fnModes = np.array(["data/AK/modes_psf/vec."+str(i+7) for i in range(3)])
mol.set_normalModeVec(fnModes)

N=100
data = []
for i in range(N):
    m = Molecule("/home/guest/Workspace/Genesis/FlexibleFitting/output/min_"+str(i)+".pdb")
    data.append(m.coords.flatten())

for i in range(50) :
    m = nma_deform(mol=mol, q=[6*i, -2*i,0])
    data.append(m.coords.flatten())
data.append(mol.coords.flatten())
data.append(target.coords.flatten())

compute_pca(data=data, length=[N, 50,1,1], labels=["Genesis trajectory", "NMA trajectory", "Reference", "Target"], save=None, n_components=2)



###############################################################################################################################

import numpy as np
from src.molecule import Molecule
import time
import autograd.numpy as npg
from autograd import elementwise_grad
from src.forcefield import *
from src.constants import *
import matplotlib.pyplot as plt
import sys

CONSTANT_TEMP = (ELEMENTARY_CHARGE) ** 2 * AVOGADRO_CONST / \
     (VACUUM_PERMITTIVITY * WATER_RELATIVE_PERMIT * ANGSTROM_TO_METER * KCAL_TO_JOULE)*2

mol = Molecule("data/AK/AK_PSF.pdb")
mol.set_forcefield(psf_file="data/AK/AK.psf", prm_file="data/toppar/par_all36_prot.prm")
mol.get_energy(verbose=True, cutoff=8.0)

psf =src.io.read_psf("data/AK/AK.psf")
prm =src.io.read_prm("data/toppar/par_all36_prot.prm")

mol = Molecule("data/P97/5ftm_psf.pdb")
mol.set_forcefield(psf_file="data/P97/5ftm.psf", prm_file="data/toppar/par_all36_prot.prm")

ep = get_excluded_pairs(forcefield=mol.forcefield)
pl = get_pairlist(coord=mol.coords, excluded_pairs=ep, cutoff=4.0, verbose=True)
invdist = get_invdist(coord=mol.coords,pairlist=pl)

t = time.time()
F, F_abs = get_autograd(params={"local": np.zeros(mol.coords.shape)}, mol=mol, potentials=["bonds", "angles", "dihedrals", "urey",
            "impropers", "vdw", "elec"], pairlist = pl, limit=500)
print(time.time()-t)
t = time.time()
F2 = get_autograd2(params={"local": np.zeros(mol.coords.shape)}, mol=mol, potentials=["bonds", "angles", "dihedrals",
            "impropers", "vdw", "elec"], pairlist = pl)["local"]
print(time.time()-t)

Fms = np.linalg.norm(F, axis=1)
plt.figure()
plt.plot(Fms)

def get_autograd_elec(coord, pairlist, forcefield):
    def get_elec(coord, pairlist, forcefield):
        invdist = get_invdist(coord, pairlist)
        return get_energy_elec(invdist, pairlist, forcefield)
    grad = elementwise_grad(get_elec, 0)
    return grad(coord, pairlist, forcefield)

def get_grad_elec(coord, invdist, pairlist, invpairlist, forcefield):
    F = np.zeros(coord.shape)
    for i in range(coord.shape[0]):
        dU = forcefield.charge[invpairlist[i]] * forcefield.charge[i] * invdist ** 3 * CONSTANT_TEMP
        dU = (coord[pairlist[:, 0]] - coord[pairlist[:, 1]]).T * dU
        F[i] -= np.sum(dU[invpairlist[i]])
        F[invpairlist[i]] += dU[:, i]
    return F
def get_grad_elec(coord, pairlist, forcefield):
    F = np.zeros(coord.shape)
    for i in range(coord.shape[0]):
        dist = np.linalg.norm(coord[i] - coord[pairlist[i]], axis=1)
    dU = forcefield.charge[pairlist[:,0]] * forcefield.charge[pairlist[:,1]] * invdist ** 3 * CONSTANT_TEMP
    dU = (coord[pairlist[:, 0]] - coord[pairlist[:, 1]]).T  * dU
    for i in range(dU.shape[0]):
        F[i] -= np.sum(dU[invpairlist[i]])
        F[invpairlist[i]] += dU[:, i]
    return dU



t = time.time()
F1 = get_autograd_elec(coord=mol.coords, forcefield=mol.forcefield, pairlist = pl)
print(time.time() - t)

t = time.time()
F2 = get_grad_elec(coord=mol.coords, pairlist=invpl, forcefield=mol.forcefield)
print(time.time() - t)

def get_excluded_pairs(forcefield):
    print("Building dic ...")
    excluded_pairs = {}
    for i in np.concatenate((forcefield.bonds, forcefield.angles[:,[0,2]])):
        if i[0] in excluded_pairs:
            excluded_pairs[i[0]].append(i[1])
        else:
            excluded_pairs[i[0]]= [i[1]]
        if i[1] in excluded_pairs:
            excluded_pairs[i[1]].append(i[0])
        else:
            excluded_pairs[i[1]]= [i[0]]
    print("Converting to array ...")
    for i in excluded_pairs:
        excluded_pairs[i] = np.array(excluded_pairs[i])
    print("Done : "+str(sys.getsizeof(excluded_pairs)/(8*1024))+" kB")
    return excluded_pairs


def get_pairlist(coord,cutoff=10.0):
    print("Building pairlist ...")
    pairlist = []
    for i in range(coord.shape[0]):
        dist = np.linalg.norm(coord[i + 1:] - coord[i], axis=1)
        idx = np.where(dist < cutoff)[0] + i + 1
        for j in idx:
            pairlist.append([i, j])
    print("Converting to array ..." + str(sys.getsizeof(pairlist) / (8 * 1024)) + " kB")
    pl_arr= np.array(pairlist)
    print("Done : " + str(sys.getsizeof(pl_arr) / (8 * 1024)) + " kB")
    return pl_arr

def get_pairlist2(coord,excluded_pairs, cutoff=10.0):
    print("Building pairlist ...")
    pairlist = []
    n_atoms=coord.shape[0]
    for i in range(n_atoms):
        dist_idx = np.setdiff1d(np.arange(i + 1, n_atoms), excluded_pairs[i])
        dist = np.linalg.norm(coord[dist_idx] - coord[i], axis=1)
        idx = np.where(dist < cutoff)[0] + i + 1
        for j in idx:
            pairlist.append([i, j])
    print("Converting to array ..." + str(sys.getsizeof(pairlist) / (8 * 1024)) + " kB")
    pl_arr= np.array(pairlist)
    print("Done : " + str(sys.getsizeof(pl_arr) / (8 * 1024)) + " kB")
    return pl_arr

def get_pairlist3(coord,cutoff=10.0):
    print("Building pairlist ...")
    pairlist = np.array([[],[]])
    for i in range(coord.shape[0]):
        dist = np.linalg.norm(coord[i + 1:] - coord[i], axis=1)
        idx = np.where(dist < cutoff)[0] + i + 1
        pairlist= np.concatenate((np.array([np.full(shape=idx.shape[0],fill_value=i,dtype=np.int), idx]),pairlist), axis=1)
    print("Done : " + str(sys.getsizeof(pairlist) / (8 * 1024)) + " kB")
    return pairlist

t=time.time()
ep = get_excluded_pairs(mol.forcefield)
print(time.time()-t)
print()

t=time.time()
pl = get_pairlist(mol.coords, cutoff=10.0)
print(time.time()-t)
print()

t=time.time()
pl2 = get_pairlist2(mol.coords, ep, cutoff=10.0)
print(time.time()-t)
print()

t=time.time()
pl3 = get_pairlist3(mol.coords, cutoff=10.0)
print(time.time()-t)
print()

import numpy as np
from src.molecule import Molecule
import time
import autograd.numpy as npg
from autograd import elementwise_grad
from src.forcefield import *
from src.constants import *
import matplotlib.pyplot as plt
from src.density import Volume
from src.viewers import chimera_fit_viewer
from src.flexible_fitting import FlexibleFitting, multiple_fitting
import sys
from src.functions import compute_pca


mol = Molecule("/home/guest/ScipionUserData/projects/BayesianFlexibleFitting/Runs/000637_ProtImportPdb/extra/AK.pdb")
mol.center()
mol.set_normalModeVec(np.array(["/home/guest/ScipionUserData/projects/BayesianFlexibleFitting/Runs/000922_FlexProtNMA/modes/vec." + str(i + 7) for i in range(3)]))
mol.allatoms2carbonalpha()
mol.set_forcefield()

init = Volume.from_coords(coord=mol.coords, size=64, voxel_size=2.0, sigma=2.0, cutoff=6.0)
params = { "biasing_factor": 100 , "global_dt":0.05, "n_iter":20, "n_warmup":10, "n_step":10}
models=[]
for i in range(20):
    vol = Volume.from_file("/home/guest/ScipionUserData/projects/BayesianFlexibleFitting/Runs/001347_FlexProtBayesianFlexibleFitting/extra/"+
                           str(i+1).zfill(5)+"_reconstructed.mrc", cutoff=6.0, voxel_size=2.0, sigma=2.0)
    vol.rescale(init)
    models.append(FlexibleFitting(init=mol, target=vol, vars=["global"], n_chain=2, params=params, verbose=0))
fits = multiple_fitting(models=models, n_chain=2, n_proc=10)

compute_pca(data=[i.res["mol"].coords.flatten() for i in fits], length=[20], n_components=2)

data = []
for i in range(20):
    mol = Molecule(
        "/home/guest/ScipionUserData/projects/BayesianFlexibleFitting/Runs/001347_FlexProtBayesianFlexibleFitting/extra/"+str(i).zfill(5)+"_fitted.pdb")
    data.append(mol.coords.flatten())

compute_pca(data=data, length=[20], n_components=2)



mol.set_forcefield(psf_file="data/AK/AK.psf", prm_file="data/toppar/par_all36_prot.prm")

T = 300
sigma = (np.ones((3, mol.n_atoms)) * np.sqrt((K_BOLTZMANN * T) /
                                                            (mol.forcefield.mass * ATOMIC_MASS_UNIT)) * 1e10).T
v = np.random.normal(0, sigma)
K =  1 / 2 * np.sum((mol.forcefield.mass*ATOMIC_MASS_UNIT)*np.square(v.T) * ANGSTROM_TO_METER**2 )
Tinst = 2 * K / (K_BOLTZMANN * 3 * mol.n_atoms)
print(Tinst)


from src.flexible_fitting import FlexibleFitting
import matplotlib.pyplot as plt
import numpy as np
from src.functions import compute_pca

N=50
CC_x = []
CC_q = []
CC_a = []

RMSD_x = []
RMSD_q = []
RMSD_a = []

data_x =[]
data_q =[]
data_a =[]
data_init=[]

for i in range(N):

    fits_x = FlexibleFitting.load("results/1ake24ake/fit_x"+str(i)+"_output.pkl")
    fits_q = FlexibleFitting.load("results/1ake24ake/fit_q"+str(i)+"_output.pkl")
    fits_a = FlexibleFitting.load("results/1ake24ake/fit_a"+str(i)+"_output.pkl")

    CC_x.append(np.mean([np.array(i["CC"]) for i in fits_x.fit], axis=0))
    CC_q.append(np.mean([np.array(i["CC"]) for i in fits_q.fit], axis=0))
    CC_a.append(np.mean([np.array(i["CC"]) for i in fits_a.fit], axis=0))

    RMSD_x.append(np.mean([np.array(i["RMSD"]) for i in fits_x.fit], axis=0))
    RMSD_q.append(np.mean([np.array(i["RMSD"]) for i in fits_q.fit], axis=0))
    RMSD_a.append(np.mean([np.array(i["RMSD"]) for i in fits_a.fit], axis=0))

    data_x.append(fits_x.res["mol"].coords.flatten())
    data_q.append(fits_q.res["mol"].coords.flatten())
    data_a.append(fits_a.res["mol"].coords.flatten())
    data_init.append(fits_a.params["target_coords"].flatten())

CC_x = np.mean(CC_x, axis=0)
CC_q = np.mean(CC_q, axis=0)
CC_a = np.mean(CC_a, axis=0)
RMSD_x= np.mean(RMSD_x, axis=0)
RMSD_q= np.mean(RMSD_q, axis=0)
RMSD_a= np.mean(RMSD_a, axis=0)

fig, ax = plt.subplots(2,1)
ax[0].plot(CC_x, label="Local")
ax[0].plot(CC_q, label="Global")
ax[0].plot(CC_a, label="Local+Global")
ax[1].plot(RMSD_x, label="Local")
ax[1].plot(RMSD_q, label="Global")
ax[1].plot(RMSD_a, label="Local+Global")
ax[0].set_ylabel("CC")
ax[0].set_xlabel("MD step")
ax[1].set_ylabel("RMSD (A)")
ax[1].set_xlabel("MD step")
# ax[0].set_xscale('log')
# ax[1].set_xscale('log')
plt.legend()

compute_pca(data=data_init+ data_x + data_q +data_a, length=[N,N,N,N], n_components=2, labels=["Reference","Local", "Global", "Local+Global"])

i=0
fit = FlexibleFitting.load("results/1ake24ake/fit_x"+str(i)+"_output.pkl")
# fit.show_3D()
target = Molecule.copy(fit.init)
target.coords = fit.params["target_coords"]
chimera_molecule_viewer([target,fit.res["mol"]])
chimera_molecule_viewer([target])

############################################################
# 1ake to 4ake
############################################################
from src.functions import compute_pca
from src.molecule import Molecule
N=200
data=[]
for i in range(N):
    mol= Molecule("/home/guest/Workspace/Genesis/FlexibleFitting/output/1ake24ake_"+str(i)+".pdb")
    data.append(mol.coords.flatten())
data.append(Molecule("data/1AKE/1ake_chainA_psf.pdb").coords.flatten())
data.append(Molecule("data/4AKE/4ake_fitted.pdb").coords.flatten())

compute_pca(data=data, length=[N,1,1,1], n_components=2)


from src.viewers import chimera_molecule_viewer
ak1 = Molecule("data/AK/AK_PSF.pdb")
ak2 = Molecule("data/4AKE/4ake_fitted.pdb")
ak2.center()

chimera_molecule_viewer([ak1,ak2 ])



#########################################################################
# Speed test GENESIS
######################################################################""


from src.molecule import Molecule
from src.viewers import *
from src.functions import get_RMSD_coords, cross_correlation, get_mol_conv
import matplotlib.pyplot as plt
from src.flexible_fitting import FlexibleFitting
from src.io import create_psf
from src.density import Volume

pdb_5ftm = Molecule("data/P97/5ftm_psf.pdb")
pdb_5ftn = Molecule("data/P97/5ftn_PSF.pdb")


res_bff = Molecule("/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/BSF/fita_all_chain2.pdb")
res_genesis = Molecule("/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/Genesis/5ftm25ftn_188.pdb")
pdb_5ftm.center()
pdb_5ftn.center()
res_bff.center()
res_genesis.center()
chimera_molecule_viewer([pdb_5ftm, pdb_5ftn, res_bff, res_genesis])

# res_genesis.center()

idx = get_mol_conv(pdb_5ftm, pdb_5ftn)
rmsd_i = get_RMSD_coords(pdb_5ftm.coords[idx[:,0]], pdb_5ftn.coords[idx[:,1]])

rmsd_i_genesis = get_RMSD_coords(res_genesis.coords, pdb_5ftm.coords)
rmsd_f_genesis = get_RMSD_coords(res_genesis.coords[idx[:,0]], pdb_5ftn.coords[idx[:,1]])
rmsd_i_bff = get_RMSD_coords(res_bff.coords, pdb_5ftm.coords)
rmsd_f_bff = get_RMSD_coords(res_bff.coords[idx[:,0]], pdb_5ftn.coords[idx[:,1]])
print(rmsd_i_genesis)
print(rmsd_f_genesis)
print(rmsd_i_bff)
print(rmsd_f_bff)

def get_cc_rmsd(N, prefix, target, size, voxel_size, cutoff, sigma):
    target.center()
    target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, cutoff=cutoff, sigma=sigma)
    rmsd=[]
    cc =[]
    for i in range(N):
        print(i)
        mol = Molecule(prefix+str(i)+".pdb")
        mol.center()
        rmsd.append(get_RMSD_coords(mol.coords, target.coords))
        vol = Volume.from_coords(coord=mol.coords, size=size, voxel_size=voxel_size, cutoff=cutoff, sigma=sigma)
        cc.append(cross_correlation(vol.data,target_density.data))
        np.save(file=prefix+"cc.npy", arr=np.array(cc))
        np.save(file=prefix+"rmsd.npy", arr=np.array(rmsd))
    return np.array(cc), np.array(rmsd)

cc, rmsd = get_cc_rmsd(N=188, prefix="/home/guest/Workspace/Paper_Frontiers/AKsynth/Genesis/AK_",
    target=Molecule("/home/guest/Workspace/Paper_Frontiers/AKsynth/AK_synth_min.pdb"), size=100, voxel_size=2.0, cutoff=6.0, sigma=2.0)

cc, rmsd = get_cc_rmsd(N=188, prefix="/home/guest/Workspace/Paper_Frontiers/AK21ake/Genesis/AK_k50000_",
    target=Molecule("data/1AKE/1ake_center.pdb"), size=100, voxel_size=2.0, cutoff=6.0, sigma=2.0)

cc, rmsd = get_cc_rmsd(N=188, prefix="/home/guest/Workspace/Paper_Frontiers/P97synth/Genesis/p97sytnh_",
    target=Molecule("data/P97/5ftm_synth_min.pdb"), size=128, voxel_size=2.0, cutoff=6.0, sigma=2.0)



fit_a = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/P97synth/fita_chain0.pkl")
fit_x = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/P97synth/fitx_chain0.pkl")
fit_q = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/P97synth/fitq_chain0.pkl")
fit_a.fit = [fit_a.fit]
fit_q.fit = [fit_q.fit]
fit_x.fit = [fit_x.fit]
cc= np.load(file="/home/guest/Workspace/Paper_Frontiers/P97synth/Genesis/p97sytnh_cc.npy")
rmsd = np.load(file="/home/guest/Workspace/Paper_Frontiers/P97synth/Genesis/p97sytnh_rmsd.npy")

fit_a = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/BSF/fita_all_chain2.pkl")
fit_a.fit = [fit_a.fit]
fit_x = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/BSF/fitx_all2_chain1.pkl")
fit_x.fit = [fit_x.fit]
fit_q = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/BSF/5ftm25ftn_noR_q_output.pkl")
cc= np.load(file="/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/Genesis/5ftm25ftn__cc.npy")
rmsd = np.load(file="/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/Genesis/5ftm25ftn__rmsd.npy")


fit_a = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/BSF/fit_a_chain0.pkl")
fit_a.fit = [fit_a.fit]
fit_x = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/BSF/fit_x_chain0.pkl")
fit_x.fit = [fit_x.fit]
fit_q = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/fit_q_dt2_output.pkl")
cc= np.load(file="/home/guest/Workspace/Paper_Frontiers/AK21ake/Genesis/AK_k50000_cc.npy")
rmsd = np.load(file="/home/guest/Workspace/Paper_Frontiers/AK21ake/Genesis/AK_k50000_rmsd.npy")


fit_a = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AKsynth/fita_output.pkl")
fit_a.fit = [fit_a.fit[0]]
fit_x = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AKsynth/fitx_output.pkl")
fit_x.fit = [fit_x.fit[1]]
fit_q = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AKsynth/fitq_output.pkl")
fit_q.fit = [fit_q.fit[1]]
cc= np.load(file="/home/guest/Workspace/Paper_Frontiers/AKsynth/Genesis/AK_k50000_cc.npy")
rmsd = np.load(file="/home/guest/Workspace/Paper_Frontiers/AKsynth/Genesis/AK_k50000_rmsd.npy")

fit_a = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/ATPase/fita_chain3.pkl")
fit_a.fit = [fit_a.fit]
fit_x = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/ATPase/fitx_chain3.pkl")
fit_x.fit = [fit_x.fit]
fit_q = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AKsynth/fitq_output.pkl")
fit_q.fit = [fit_q.fit[1]]
cc= np.load(file="/home/guest/Workspace/Paper_Frontiers/AKsynth/Genesis/AK_k50000_cc.npy")
rmsd = np.load(file="/home/guest/Workspace/Paper_Frontiers/AKsynth/Genesis/AK_k50000_rmsd.npy")

with plt.style.context("bmh"):
    fig, ax = plt.subplots(1,2, figsize = (10,3))
    cc_init =np.mean([np.array(i["CC"]) for i in fit_x.fit], axis=0)[0]
    rmsd_init =np.mean([np.array(i["RMSD"]) for i in fit_x.fit], axis=0)[0]
    ax[0].plot(np.arange(len(cc)+1)*100,[cc_init] + list(cc), label="Genesis", c="tab:orange")
    ax[1].plot(np.arange(len(cc)+1)*100,[rmsd_init] + list(rmsd), label="Genesis", c="tab:orange")
    ax[0].plot(np.mean([np.array(i["CC"]) for i in fit_x.fit], axis=0), label="Local", c="tab:green")
    ax[0].plot(np.mean([np.array(i["CC"]) for i in fit_q.fit], axis=0), label="Global", c="tab:blue")
    ax[0].plot(np.mean([np.array(i["CC"]) for i in fit_a.fit], axis=0), label="Local+Global", c="tab:red")
    ax[1].plot(np.mean([np.array(i["RMSD"]) for i in fit_x.fit], axis=0), label="Local", c="tab:green")
    ax[1].plot(np.mean([np.array(i["RMSD"]) for i in fit_q.fit], axis=0), label="Global", c="tab:blue")
    ax[1].plot(np.mean([np.array(i["RMSD"]) for i in fit_a.fit], axis=0), label="Local+Global", c="tab:red")
    ax[0].set_xlabel("MD step")
    ax[0].set_ylabel("CC")
    ax[0].set_title("Cross correlation")
    ax[0].legend(loc='lower right')
    ax[1].set_xlabel("MD step")
    ax[1].set_ylabel("RMSD (A)")
    ax[1].set_title("Root Mean Square Deviation")
    N=1000
    ax[0].set_xlim(-N/10,N + N/10)
    ax[1].set_xlim(-N/10,N + N/10)
    fig.tight_layout()






fit_a.show_3D()

#########################################################################
# p97 all atoms
######################################################################""

from src.flexible_fitting import FlexibleFitting
import matplotlib.pyplot as plt

fitx = FlexibleFitting.load(file="results/P97/5ftm25ftn_noR_x_output.pkl")
fitq = FlexibleFitting.load(file="results/P97/5ftm25ftn_noR_q_output.pkl")
fita = FlexibleFitting.load(file="results/P97/5ftm25ftn_noR_a_output.pkl")

CC_x = np.mean([i["CC"] for i in fitx.fit], axis=0)
CC_q = np.mean([i["CC"] for i in fitq.fit], axis=0)
CC_a = np.mean([i["CC"] for i in fita.fit], axis=0)

RMSD_x= np.mean([i["RMSD"] for i in fitx.fit], axis=0)
RMSD_q= np.mean([i["RMSD"] for i in fitq.fit], axis=0)
RMSD_a= np.mean([i["RMSD"] for i in fita.fit], axis=0)

fig, ax = plt.subplots(1,1, figsize=(6,3))
ax.plot(CC_x, label="Local")
ax.plot(CC_q, label="Global")
ax.plot(CC_a, label="Local+Global")
ax.set_ylabel("CC")
ax.set_xlabel("MD step")

# ax[0].set_xscale('log')
# ax[1].set_xscale('log')
plt.legend()
fig.tight_layout()



from src.viewers import ramachandran_viewer

ramachandran_viewer("/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/5ftm25ftn_noR_a_output.pdb")
ramachandran_viewer("/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/BSF/fita_all_chain2.pdb")
ramachandran_viewer("/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/Genesis/5ftm25ftn_188.pdb")
ramachandran_viewer("data/P97/5ftm_psf.pdb")

ramachandran_viewer("data/AK/AK_PSF.pdb")
ramachandran_viewer("/home/guest/Workspace/Paper_Frontiers/AK21ake/fit_a_dt2_chain0_min.pdb")
ramachandran_viewer("/home/guest/Workspace/Paper_Frontiers/AK21ake/fit_a_chain2.pdb")
ramachandran_viewer("/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/Genesis/5ftm25ftn_188.pdb")







import seaborn as sbn



fit_a = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AKsynth/fita_output.pkl")
fit_a.fit = [fit_a.fit[0]]
fit_x = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AKsynth/fitx_output.pkl")
fit_x.fit = [fit_x.fit[1]]
fit_q = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AKsynth/fitq_output.pkl")
fit_q.fit = [fit_q.fit[1]]
cc= np.load(file="/home/guest/Workspace/Paper_Frontiers/AKsynth/Genesis/AK_k50000_cc.npy")
rmsd = np.load(file="/home/guest/Workspace/Paper_Frontiers/AKsynth/Genesis/AK_k50000_rmsd.npy")


with plt.style.context("bmh"):
    fig, ax = plt.subplots(1,2, figsize = (10,3))
    cc_init =np.mean([np.array(i["CC"]) for i in fit_x.fit], axis=0)[0]
    rmsd_init =np.mean([np.array(i["RMSD"]) for i in fit_x.fit], axis=0)[0]
    ax[0].plot(np.arange(len(cc)+1)*100,[cc_init] + list(cc), label="Genesis",                 c="tab:orange")
    ax[1].plot(np.arange(len(cc)+1)*100,[rmsd_init] + list(rmsd), label="Genesis",             c="tab:orange")
    ax[0].plot(np.mean([np.array(i["CC"]) for i in fit_x.fit], axis=0), label="Local",         c="tab:green")
    ax[0].plot(np.mean([np.array(i["CC"]) for i in fit_q.fit], axis=0), label="Global",        c="tab:blue")
    ax[0].plot(np.mean([np.array(i["CC"]) for i in fit_a.fit], axis=0), label="Local+Global",  c="tab:red")
    ax[1].plot(np.mean([np.array(i["RMSD"]) for i in fit_x.fit], axis=0), label="Local",       c="tab:green")
    ax[1].plot(np.mean([np.array(i["RMSD"]) for i in fit_q.fit], axis=0), label="Global",      c="tab:blue")
    ax[1].plot(np.mean([np.array(i["RMSD"]) for i in fit_a.fit], axis=0), label="Local+Global",c="tab:red")
    ax[0].set_xlabel("MD step")
    ax[0].set_ylabel("CC")
    ax[0].set_title("Cross correlation")
    ax[0].legend(loc='lower right')
    ax[1].set_xlabel("MD step")
    ax[1].set_ylabel("RMSD (A)")
    ax[1].set_title("Root Mean Square Deviation")
    N=1000
    ax[0].set_xlim(-N/10,N + N/10)
    ax[1].set_xlim(-N/10,N + N/10)
    fig.tight_layout()