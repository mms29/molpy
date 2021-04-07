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

mol = Molecule("data/AK/AK_PSF.pdb")
mol.set_forcefield(psf_file="data/AK/AK.psf", prm_file="data/toppar/par_all36_prot.prm")
mol.get_energy(verbose=True)

mol = Molecule("data/P97/5ftm_psf.pdb")
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