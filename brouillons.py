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
from src.functions import get_RMSD_coords, get_mol_conv
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

def get_cc_rmsd(N, prefix, target, size, voxel_size, cutoff, sigma, step=1, test_idx=False):
    target.center()
    target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, cutoff=cutoff, sigma=sigma)
    rmsd=[]
    cc =[]
    idx = None
    for i in range(0,N,step):
        print(i)
        mol = Molecule(prefix+str(i)+".pdb")
        mol.center()

        if test_idx:
            if idx is None:
                idx = src.functions.get_mol_conv(mol, target)
            rmsd.append(get_RMSD_coords(mol.coords[idx[:,0]], target.coords[idx[:,1]]))
        else:
            rmsd.append(get_RMSD_coords(mol.coords, target.coords))
        vol = Volume.from_coords(coord=mol.coords, size=size, voxel_size=voxel_size, cutoff=cutoff, sigma=sigma)
        cc.append(src.density.get_CC(vol.data,target_density.data))
        np.save(file=prefix+"cc.npy", arr=np.array(cc))
        np.save(file=prefix+"rmsd.npy", arr=np.array(rmsd))
    return np.array(cc), np.array(rmsd)

cc, rmsd = get_cc_rmsd(N=188, prefix="/home/guest/Workspace/Paper_Frontiers/AKsynth/Genesis/AK_",
    target=Molecule("/home/guest/Workspace/Paper_Frontiers/AKsynth/AK_synth_min.pdb"), size=100, voxel_size=2.0, cutoff=6.0, sigma=2.0)

cc, rmsd = get_cc_rmsd(N=100, prefix="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000185_FlexProtGenesisFit/extra/AK_K10000_",
    target=Molecule("data/1AKE/1ake_center.pdb"), size=100, voxel_size=2.0, cutoff=6.0, sigma=2.0, test_idx=False)

cc, rmsd = get_cc_rmsd(N=188, prefix="/home/guest/Workspace/Paper_Frontiers/P97synth/Genesis/p97sytnh_",
    target=Molecule("data/P97/5ftm_synth_min.pdb"), size=128, voxel_size=2.0, cutoff=6.0, sigma=2.0)

cc, rmsd = get_cc_rmsd(N=188, prefix="/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/Genesis/5ftm25ftn_",
    target=Molecule("data/P97/5ftn_PSF.pdb"), size=200, voxel_size=2.0, cutoff=6.0, sigma=2.0, step=1)



fit_a = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/P97synth/fita_chain0.pkl")
fit_x = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/P97synth/fitx_chain0.pkl")
fit_q = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/P97synth/fitq_chain0.pkl")
fit_a.fit = [fit_a.fit]
fit_q.fit = [fit_q.fit]
fit_x.fit = [fit_x.fit]
cc= np.load(file="/home/guest/Workspace/Paper_Frontiers/P97synth/Genesis/p97sytnh_cc.npy")
rmsd = np.load(file="/home/guest/Workspace/Paper_Frontiers/P97synth/Genesis/p97sytnh_rmsd.npy")

fit_a = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/BSF/fita_final_chain0.pkl")
fit_a.fit = [fit_a.fit]
fit_x = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/BSF/fitx_all2_chain2.pkl")
fit_x.fit = [fit_x.fit]
fit_q = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/BSF/fitq_final_chain0.pkl")
fit_q.fit = [fit_q.fit]
cc= np.load(file="/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/Genesis/5ftm25ftn_cc.npy")
rmsd = np.load(file="/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/Genesis/5ftm25ftn_rmsd.npy")


fit_a = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/complete/fita_cc3_chain0.pkl")
fit_a.fit = [fit_a.fit]
fit_x = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/complete/fitx_cc3_chain0.pkl")
fit_x.fit = [fit_x.fit]
fit_q = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/complete/fitq_cc3_chain0.pkl")
fit_q.fit = [fit_q.fit]
# fit_q = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/fit_q_dt2_output.pkl")
cc= np.load(file="/home/guest/Workspace/Paper_Frontiers/AK21ake/Genesis/AK_K10000_cc.npy")
rmsd = np.load(file="/home/guest/Workspace/Paper_Frontiers/AK21ake/Genesis/AK_K10000_rmsd.npy")


fit_a = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AKsynth/fita_output.pkl")
fit_a.fit = [fit_a.fit[0]]
fit_x = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AKsynth/fitx_output.pkl")
fit_x.fit = [fit_x.fit[1]]
fit_q = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AKsynth/fitq_output.pkl")
fit_q.fit = [fit_q.fit[1]]
cc= np.load(file="/home/guest/Workspace/Paper_Frontiers/AKsynth/Genesis/AK_k50000_cc.npy")
rmsd = np.load(file="/home/guest/Workspace/Paper_Frontiers/AKsynth/Genesis/AK_k50000_rmsd.npy")

fit_a = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/RF2/fita_cc_chain0.pkl")
fit_a.fit = [fit_a.fit]
fit_x = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/ATPase/fitx_chain3.pkl")
fit_x.fit = [fit_x.fit]
fit_q = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AKsynth/fitq_output.pkl")
fit_q.fit = [fit_q.fit[1]]
cc= np.load(file="/home/guest/Workspace/Paper_Frontiers/AKsynth/Genesis/AK_k50000_cc.npy")
rmsd = np.load(file="/home/guest/Workspace/Paper_Frontiers/AKsynth/Genesis/AK_k50000_rmsd.npy")

with plt.style.context("bmh"):
    fig, ax = plt.subplots(1,2, figsize = (10,2.7))
    cc_init =np.mean([np.array(i["CC"]) for i in fit_x.fit], axis=0)[0]
    rmsd_init =np.mean([np.array(i["RMSD"]) for i in fit_x.fit], axis=0)[0]
    ax[0].plot(np.mean([np.array(i["CC"]) for i in fit_a.fit], axis=0), label=r"$\Delta \mathbf{r}_{Local}+\Delta \mathbf{r}_{global}$", c="tab:red")
    ax[1].plot(np.mean([np.array(i["RMSD"]) for i in fit_a.fit], axis=0), label=r"$\Delta \mathbf{r}_{Local}+\Delta \mathbf{r}_{global}$", c="tab:red")
    ax[0].plot(np.mean([np.array(i["CC"]) for i in fit_x.fit], axis=0), label=r"$\Delta \mathbf{r}_{Local}$", c="tab:green")
    ax[1].plot(np.mean([np.array(i["RMSD"]) for i in fit_x.fit], axis=0), label=r"$\Delta \mathbf{r}_{Local}$", c="tab:green")
    ax[0].plot(np.mean([np.array(i["CC"]) for i in fit_q.fit], axis=0)[::2], label=r"$\Delta \mathbf{r}_{global}$", c="tab:blue")
    ax[1].plot(np.mean([np.array(i["RMSD"]) for i in fit_q.fit], axis=0)[::2], label=r"$\Delta \mathbf{r}_{global}$", c="tab:blue")
    # ax[0].plot(np.arange(len(cc)+1)*100,[cc_init] + list(cc), label="Genesis", c="tab:orange")
    # ax[1].plot(np.arange(len(cc)+1)*100,[rmsd_init] + list(rmsd), label="Genesis", c="tab:orange")
    ax[0].set_xlabel("HMC step")
    ax[0].set_ylabel("CC")
    ax[1].legend(loc='lower right')
    ax[1].set_xlabel("HMC step")
    ax[1].set_ylabel("RMSD (A)")
    N=1000
    ax[0].set_xlim(-N/10,1000)
    ax[1].set_xlim(-N/10,1000)
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

ramachandran_viewer("data/RF2/1gqe_PSF.pdb")
ramachandran_viewer("/home/guest/Workspace/Paper_Frontiers/RF2/fita_chain0.pdb")

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









#########################################################################
# RF2
######################################################################""


from src.molecule import Molecule
from src.viewers import *
from src.functions import get_RMSD_coords, cross_correlation, get_mol_conv
import matplotlib.pyplot as plt
from src.flexible_fitting import FlexibleFitting
from src.io import create_psf
from src.density import Volume

mol1 = Molecule("data/RF2/1n0u.pdb")
mol2 = Molecule("data/RF2/1n0vC.pdb")


chimera_molecule_viewer([mol1,mol2])

def check(F):
    F["a"] = 2

F = {"a":1}
check(F)












from src.flexible_fitting import *
from src.molecule import Molecule
from src.density import Volume
from src.functions import get_RMSD_coords


ak = Molecule("tests/tests_data/input/AK/AK.pdb")
ak.center()
ak.allatoms2carbonalpha()
ak.set_forcefield()

target = Volume.from_file(file="tests/tests_data/input/AK/ak_nma_carbonalpha.mrc", voxel_size=1.5, sigma=2, cutoff=6)
params = {"biasing_factor": 8000,
          "local_dt": 2 * 1e-15,
          "n_iter": 50,
          "n_warmup": 40,
          "gradient":"CC"}
fit = FlexibleFitting(init=ak, target=target, vars=["local"], params=params, n_chain=4, verbose=2)
fit.HMC_chain()

dx= np.random.normal(1,2,ak.coords.shape)
init = Volume.from_coords(coord=ak.coords, size=64, voxel_size=1.5, sigma=2.0, cutoff=6.0)

a = src.density.get_gradient_CC(mol=ak, psim= init.data, pexp=target, params={"local" : dx})["local"]
b = src.density.get_gradient_LS(mol=ak, psim= init.data, pexp=target, params={"local" : dx})["local"]

np.mean([np.dot(a[i].T, b[i])/(np.linalg.norm(a[i])*np.linalg.norm(b[i])) for i in range(len(a))])



from src.flexible_fitting import FlexibleFitting
import matplotlib.pyplot as plt
import numpy as np
from src.molecule import Molecule
from src.density import Volume
from src.functions import *
import src.functions
import matplotlib.pylab as pl

ccs=[]
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_cc0_chain0.pkl"))
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_cc1_chain0.pkl"))
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_cc3_chain0.pkl"))
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_cc4_chain0.pkl"))
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_cc5_chain0.pkl"))

lss=[]
lss.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_ls0_chain0.pkl"))
lss.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_ls1_chain0.pkl"))
# lss.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_ls3_chain0.pkl"))
lss.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_ls4_chain0.pkl"))
lss.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_ls5_chain0.pkl"))

prefix = ["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001170_FlexProtGenesisFit/extra/run",
          "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001296_FlexProtGenesisFit/extra/run",
          "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000185_FlexProtGenesisFit/extra/AK_K10000_",
          "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001338_FlexProtGenesisFit/extra/run",
          "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001254_FlexProtGenesisFit/extra/run",
          "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001209_FlexProtGenesisFit/extra/run"]
cc=[]
rmsd= []
for i in prefix :
    cc.append(np.load(file=i + "cc.npy"))
    rmsd.append(np.load(file=i + "rmsd.npy"))
lgen=["5000", "7500", "10000", "17500", "25000","50000"]


# file = "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001422_FlexProtGenesisFit/extra/run"
# cc, rmsd = get_cc_rmsd(N=100, prefix=file,
#     target=Molecule("data/1AKE/1ake_center.pdb"), size=64, voxel_size=2.0, cutoff=6.0, sigma=2.0, step=1, test_idx=False)



fig, ax = plt.subplots(2,1)
for i in range(len(ccs)):
    ax[0].plot(ccs[i].fit["CC"], label=ccs[i].params["biasing_factor"], c=cccs[i])
    ax[1].plot(ccs[i].fit["RMSD"],c=cccs[i])
for i in range(len(lss)):
    ax[0].plot(lss[i].fit["CC"], label=lss[i].params["biasing_factor"], c=clss[i])
    ax[1].plot(lss[i].fit["RMSD"],c=clss[i])
for i in range(len(prefix)):
    ax[0].plot((np.arange(len(cc[i]))+1)*100, cc[i], label=lgen[i], c=cgen[i])
    ax[1].plot((np.arange(len(rmsd[i]))+1)*100, rmsd[i], c=cgen[i])
# N=700
# ax[0].set_xlim(-N/10,N + N/10)
# ax[1].set_xlim(-N/10,N + N/10)
fig.legend()
ax[0].set_xlabel("MD step")
ax[0].set_ylabel("CC")
ax[0].set_title("Cross correlation")
# ax[0].legend(loc='lower right')
ax[1].set_xlabel("MD step")
ax[1].set_ylabel("RMSD (A)")
ax[1].set_title("Root Mean Square Deviation")
fig.tight_layout()

######## ATPase




from src.flexible_fitting import FlexibleFitting
import matplotlib.pyplot as plt
import numpy as np
from src.molecule import Molecule
from src.density import Volume
from src.functions import *
import src.functions
import matplotlib.pylab as pl
#
file = "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001464_FlexProtGenesisFit/extra/run"
cc, rmsd = get_cc_rmsd(N=100, prefix=file,
    target=Molecule("data/1AKE/1ake_center.pdb"), size=100, voxel_size=2.0, cutoff=10.0, sigma=2.0, step=2, test_idx=False)

prefix = ["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001380_FlexProtGenesisFit/extra/run"]
cc=[]
rmsd= []
for i in prefix :
    cc.append(np.load(file=i + "cc.npy"))
    rmsd.append(np.load(file=i + "rmsd.npy"))

plt.plot(cc)
plt.plot(rmsd)









#AK
fit1 = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_cc3_modes_chain0.pkl")
fit2 = FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_cc3_chain0.pkl")
cc = (np.load(file="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007321_FlexProtGenesisFit/extra/run0_cc.npy"))
rmsd= (np.load(file="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007321_FlexProtGenesisFit/extra/run0_rmsd.npy"))
fig, ax = plt.subplots(1,2, figsize=(8,3))
ax[0].plot(np.arange(10)*500,np.array(fit2.fit["CC"])[:5000:500], "d-", color="tab:red")
ax[1].plot(np.arange(10)*500,np.array(fit2.fit["RMSD"])[:5000:500], "d-",label="HMC/NMA combined", color="tab:red")
cct =  np.array([fit1.fit["CC"][0]] + list(cc)) [:50:5]
rmsdt= np.array([fit1.fit["RMSD"][0]] + list(rmsd))[:50:5]
ax[0].plot(np.arange(10)*500, cct , "o-", color="tab:blue")
ax[1].plot(np.arange(10)*500,  rmsdt, "o-",label="HMC only", color="tab:blue")

ax[0].set_xlabel("HMC step")
ax[0].set_ylabel("CC")
ax[1].legend(loc='upper right')
ax[1].set_xlabel("HMC step")
ax[1].set_ylabel("RMSD (A)")
fig.tight_layout()
fig.savefig("results/test.png", dpi=1000)

###################################################################################################################
###################################################################################################################
###################################################################################################################


ccs=[]
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fitx_cc0_chain0.pkl"))
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fitx_cc1_chain0.pkl"))
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fitx_cc3_chain0.pkl"))
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fitx_cc4_chain0.pkl"))
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fitx_cc5_chain0.pkl"))
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fitx_cc6_chain0.pkl"))
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fitx_cc7_chain0.pkl"))
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fitx_cc8_chain0.pkl"))
ccs.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fitx_cc9_chain0.pkl"))

prefix = ["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001422_FlexProtGenesisFit/extra/run",]
cc=[]
rmsd= []
for i in prefix :
    cc.append(np.load(file=i + "cc.npy"))
    rmsd.append(np.load(file=i + "rmsd.npy"))

cccs = pl.cm.Reds(np.linspace(0.5,1,len(ccs)))
# clss = pl.cm.Blues(np.linspace(0.5,1,len(lss)))
cgen = pl.cm.Greens(np.linspace(0.5,1,len(cc)))


fig, ax = plt.subplots(2,1)
for i in range(len(ccs)):
    ax[0].plot(ccs[i].fit["CC"], label=ccs[i].params["biasing_factor"], c=cccs[i])
    ax[1].plot(ccs[i].fit["RMSD"],c=cccs[i])
for i in range(len(prefix)):
    ax[0].plot((np.arange(len(cc[i]))+1)*100, cc[i], label="1000", c=cgen[i])
    ax[1].plot((np.arange(len(rmsd[i]))+1)*100, rmsd[i], c=cgen[i])
N=10000
ax[0].set_xlim(-N/10,N + N/10)
ax[1].set_xlim(-N/10,N + N/10)
fig.legend()












############################" modes/no modes Fabs
fits=[]
fits.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_modes_chain0.pkl"))
# fits.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_nomodes_chain0.pkl"))
fits.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_modes_hmc1_chain0.pkl"))
# fits.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_nomodes_hmc1_chain0.pkl"))
fits.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_modes_hmc2_chain0.pkl"))
# fits.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_nomodes_hmc2_chain0.pkl"))
leg = ["1000 step","100 step", "10 step"]
fig, ax = plt.subplots(2,1)
for i in range(len(fits)):
    ax[0].plot(fits[i].fit["CC"], label = leg[i])
    ax[1].plot(fits[i].fit["RMSD"])
N=1000
ax[0].set_xlim(-N/10,N + N/10)
ax[1].set_xlim(-N/10,N + N/10)
ax[0].set_xlabel("MD step")
ax[0].set_ylabel("CC")
ax[0].set_title("Cross correlation")
ax[0].legend(loc='lower right')
ax[1].set_xlabel("MD step")
ax[1].set_ylabel("RMSD (A)")
ax[1].set_title("Root Mean Square Deviation")
fig.tight_layout()

fits=[]
fits.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_modes_Fabs_chain0.pkl"))
fits.append(FlexibleFitting.load("/home/guest/Workspace/Paper_Frontiers/AK21ake/new/fita_nomodes_Fabs_chain0.pkl"))
fig, ax = plt.subplots(2,1)
for i in range(len(fits)):
    ax[0].plot(np.abs(fits[i].fit["local_Fabs2"]) *(fits[i].params["local_dt"]**2))
    ax[1].plot(np.abs(fits[i].fit["global_Fabs2"])*(fits[i].params["global_dt"]**2))



temperature = 300
local_sigma = (np.ones((3, init.n_atoms)) *  np.sqrt((K_BOLTZMANN * temperature) /
                    (init.forcefield.mass * ATOMIC_MASS_UNIT)) * ANGSTROM_TO_METER**-1).T
local_v = np.random.normal(0, local_sigma, init.coords.shape) # A*s-1
local_v = local_v

local_v = np.random.normal(0, local_sigma, init.coords.shape) # A*s-1
global_v = np.zeros(4)
for i in range(init.n_atoms):
    global_v += np.dot(init.normalModeVec[i], local_v[i] )


K = 1 / 2 * np.sum((init.forcefield.mass * ATOMIC_MASS_UNIT) * np.square(local_v + np.dot(global_v, init.normalModeVec)).T)
K *= ANGSTROM_TO_METER ** 2 * (AVOGADRO_CONST / KCAL_TO_JOULE)  # kg * A2 * s-2 -> kcal * mol-1
T = 2 * K*(KCAL_TO_JOULE/AVOGADRO_CONST ) / (K_BOLTZMANN * 3 * init.n_atoms)

local_v *= temperature/T

dU_biased = src.density.get_gradient_CC(mol=init, psim=init_density.data, pexp=target_density,
                                    params={"local": np.zeros(init.coords.shape), "global" : np.zeros(4)},
                                        normalModeVec=init.normalModeVec)
F_local = -(dU_biased["local"])
F_global = -(dU_biased["global"])
np.dot(F_global, init.normalModeVec)

F_local = (F_local.T * (1 / (init.forcefield.mass * ATOMIC_MASS_UNIT))).T  # Force -> acceleration
F_local *= (KCAL_TO_JOULE / AVOGADRO_CONST)  # kcal/mol -> Joule
F_local *= ANGSTROM_TO_METER ** -2  # kg * m2 * s-2 -> kg * A2 * s-2


mq = np.zeros(4)
for i in range(init.n_atoms):
    mq += np.linalg.norm(init.normalModeVec[i], axis=1) * init.forcefield.mass[i]

F_global = (F_global.T * (1 / (mq * ATOMIC_MASS_UNIT))).T  # Force -> acceleration
F_global *= (KCAL_TO_JOULE / AVOGADRO_CONST)  # kcal/mol -> Joule
F_global *= ANGSTROM_TO_METER ** -2  # kg * m2 * s-2 -> kg * A2 * s-2






from src.molecule import Molecule
from src.density import Volume
from src.density import get_gradient_CC
import numpy as np
import matplotlib.pyplot as plt
from src.forcefield import get_autograd, get_pairlist

# import PDB
init =Molecule("/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000079_FlexProtGenesisMin/extra/output.pdb")
init.center()
fnModes = np.array(["data/AK/modes_psf/vec."+str(i+7) for i in range(4)])
init.set_normalModeVec(fnModes)
init.set_forcefield(psf_file="data/AK/AK.psf", prm_file= "data/toppar/par_all36_prot.prm")
target = Volume.from_file(file="data/1AKE/1ake_center.mrc", sigma=2.0, cutoff=30.0, voxel_size=2.0)
init_density = Volume.from_coords(coord=init.coords, size=target.size, voxel_size=target.voxel_size, sigma=target.sigma, cutoff=target.cutoff)

pl = get_pairlist(coord=init.coords, excluded_pairs=init.forcefield.excluded_pairs, cutoff=20.0)
f1 = get_autograd(params={"local": np.zeros(init.coords.shape)}, mol=init, potentials=["elec"], pairlist=pl)["local"]
f1[np.where(f1==0)] = 0.0001
# f1 = src.density.get_gradient_CC(mol=init, psim=init_density.data, pexp=target, params={"local": np.zeros(init.coords.shape)})["local"]
f= get_autograd(params={"local": np.zeros(init.coords.shape), "global": np.zeros(4)}, mol=init, potentials=["bonds", "angles", "dihedrals", "impropers", "urey", "vdw", "elec"],
                  normalModeVec=init.normalModeVec, pairlist=pl)
fq= f["global"]
fx= f["local"]
fq2 = np.zeros(fq.shape)
for i in range(init.n_atoms):
    fq2 += np.dot(init.normalModeVec[i], fx[i])
print(fq-fq2)

fq = src.density.get_gradient_CC(mol=init, psim=init_density.data, pexp=target, params={"local": np.zeros(init.coords.shape),
                                 "global": np.zeros(4)}, normalModeVec = init.normalModeVec)["global"]
# f1*=10000
f2 = np.loadtxt("/opt/genesis-1.4.0/force.txt")[:f1.shape[0]]


print(np.mean(np.linalg.norm(f1+f2, axis=1)))
print(np.mean([np.dot(f1[i], f2[i])/(np.linalg.norm(f1[i])*np.linalg.norm(f2[i])) for i in range(f1.shape[0])]))

Fabs1 = np.linalg.norm(f1, axis=1)
Fabs2 = np.linalg.norm(f2, axis=1)
plt.figure()
plt.plot(Fabs1, label="1")
plt.plot(Fabs2, label="2")
plt.legend()


def show_cc_rmsd_old(protocol_list, length, labels=None, period=100, step=10, init_cc=0.7, init_rmsd=10.0,
                 fvar=10.0, capthick=10.0, capsize=10.0, elinewidth=1.0, figsize=(10,5), dt=0.002,
                 colors=["tab:blue", "tab:red", "tab:green", "tab:orange",
                         "tab:brown", "tab:olive", "tab:pink", "tab:green", "tab:cyan"],
                 fmts = ["o", "d", "v", "^", "p", "*", "s","x"], img = None):
    if labels is None:
        labels = ["#"+str(i) for i in range(len(protocol_list))]
    if img is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i in range(len(protocol_list)):
        cc = []
        rmsd = []
        for j in range(length[i]):
            cc.append([init_cc] + list(np.load(protocol_list[i]+ "/extra/run"+str(j)+"_cc.npy")))
            rmsd.append([init_rmsd] + list(np.load(protocol_list[i] +"/extra/run"+str(j)+"_rmsd.npy")))
            t = np.arange(len(cc[-1])) * period * dt
        ax[0].errorbar(x=t[::step], y=np.mean(cc, axis=0)[::step],  yerr=np.var(cc, axis=0)[::step]*fvar,
                       label=labels[i], color=colors[i], fmt=fmts[i],
                       capthick=capthick, capsize=capsize,elinewidth=elinewidth)
        ax[1].errorbar(x=t[::step], y=np.mean(rmsd, axis=0)[::step],yerr=np.var(rmsd, axis=0)[::step]*fvar,
                        color=colors[i], fmt=fmts[i],
                       capthick=capthick, capsize=capsize,elinewidth=elinewidth)
        ax[0].plot(t, np.mean(cc, axis=0), "-",color=colors[i])
        ax[1].plot(t, np.mean(rmsd, axis=0), "-",color=colors[i])
        ax[0].set_xlabel("Simulation Time (ps)")
        ax[0].set_ylabel("CC")
        ax[0].set_title("Correlation Coefficient")
        ax[1].set_xlabel("Simulation Time (ps)")
        ax[1].set_ylabel("RMSD (A)")
        ax[1].set_title("Root Mean Square Deviation")
        if img is not None:
            ax[2].imshow(mpimg.imread(img))
            ax[2].axis('off')
            ax[2].set_title("3D structures")
        fig.tight_layout()
    handles, labels = ax[0].get_legend_handles_labels()
    if img is not None:
        handles.append(mpatches.Patch(color='yellow', label='Initial structure'))
    fig.legend(handles = handles, loc='lower right')
    return fig



from src.flexible_fitting import FlexibleFitting
import matplotlib.pyplot as plt
import numpy as np
from src.molecule import Molecule
from src.density import Volume
from src.functions import *
import src.functions
from src.viewers import *
import matplotlib.pylab as pl
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
# cc, rmsd = get_cc_rmsd(N=1000, prefix="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003471_FlexProtGenesisFit/extra/run0_",
#                        target=Molecule("/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003631_FlexProtGeneratePSF/extra/output.pdb"),
#                        size=128, voxel_size=1.5, cutoff=6.0, sigma=2.0, step=10, test_idx=True)
# cc, rmsd = get_cc_rmsd(N=1000, prefix="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003670_FlexProtGenesisFit/extra/run0_",
#                        target=Molecule("/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003631_FlexProtGeneratePSF/extra/output.pdb"),
#                        size=128, voxel_size=1.5, cutoff=6.0, sigma=2.0, step=10, test_idx=True)
#

def show_cc_rmsd(protocol_list, length, labels=None, period=100, step=10, init_cc=0.7, init_rmsd=10.0,
                 fvar=10.0, capthick=10.0, capsize=10.0, elinewidth=1.0, figsize=(10,5), dt=0.002,
                 colors=["tab:blue", "tab:red", "tab:green", "tab:orange",
                         "tab:brown", "tab:olive", "tab:pink", "tab:green", "tab:cyan"],
                 fmts = ["o", "d", "v", "^", "p", "*", "s","x"], img = None, xlim= (0,100)):
    if labels is None:
        labels = ["#"+str(i) for i in range(len(protocol_list))]
    if img is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i in range(len(protocol_list)):
        cc = []
        rmsd = []
        for j in range(length[i]):
            cc.append([init_cc] + list(np.load(protocol_list[i]+ "/extra/run"+str(j)+"_cc.npy")))
            rmsd.append([init_rmsd] + list(np.load(protocol_list[i] +"/extra/run"+str(j)+"_rmsd.npy")))
            t = np.arange(len(cc[-1])) * period * dt
        step = len(cc[-1]) // 10
        ax[0].errorbar(x=t[::step], y=np.mean(cc, axis=0)[::step]-1,  yerr=np.std(cc, axis=0)[::step]*fvar,
                       label=labels[i], color=colors[i], fmt=fmts[i],
                       capthick=capthick, capsize=capsize,elinewidth=elinewidth)
        ax[1].errorbar(x=t[::step], y=np.mean(rmsd, axis=0)[::step]-1,yerr=np.std(rmsd, axis=0)[::step]*fvar,
                        color=colors[i], fmt=fmts[i],
                       capthick=capthick, capsize=capsize,elinewidth=elinewidth)
        ax[0].plot(t, np.mean(cc, axis=0)-1, "-",color=colors[i])
        ax[1].plot(t, np.mean(rmsd, axis=0)-1, "-",color=colors[i])
        ax[0].set_xlabel("Simulation Time (ps)")
        ax[0].set_ylabel("CC")
        ax[0].set_title("Correlation Coefficient")
        ax[1].set_xlabel("Simulation Time (ps)")
        ax[1].set_ylabel("RMSD (A)")
        ax[1].set_title("Root Mean Square Deviation")
        ax[1].set_xlim(xlim[0], xlim[1])
        if img is not None:
            ax[2].imshow(mpimg.imread(img))
            ax[2].axis('off')
            ax[2].set_title("3D structures")
        fig.tight_layout()
    handles, labels = ax[0].get_legend_handles_labels()
    if img is not None:
        handles.append(mpatches.Patch(color='yellow', label='Initial structure'))
    fig.legend(handles = handles, loc='lower right')
    return fig



ak = show_cc_rmsd([
    "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007321_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007381_FlexProtGenesisFit"],
             length=[5,5], labels=["no modes", "Modes 7-9"], step=5, period=100, init_cc=0.75,
             init_rmsd=8.12, fvar=1, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002,
            img = "/home/guest/Workspace/Paper_Frontiers/screenschots/ak.png" )
#chimera /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007321_FlexProtGenesisFit/extra/target.sit /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007321_FlexProtGenesisFit/extra/run0_.pdb /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007381_FlexProtGenesisFit/extra/run0_.pdb /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006894_ProtImportPdb/extra/4ake.pdb

f2 = show_cc_rmsd(["/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000380_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000819_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000890_FlexProtGenesisFit",
              ],
             length=[1,1,1], labels=["Mode 7-9", "Mode 10-13", "No modes"], step=25, period=200, init_cc=0.764,
             init_rmsd=11.24, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)

p97 = show_cc_rmsd(["/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/new/local",
              "/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/new/global", ],
             length=[1,1,1], labels=["No modes","Mode 7-10"], step=5, period=10000, init_cc=0.764,
             init_rmsd=11.24, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)

p97 = show_cc_rmsd(["/home/guest/ScipionUserData/projects/p97/Runs/000461_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/p97/Runs/000521_FlexProtGenesisFit", ],
             length=[1,1,1], labels=["No modes","Mode 7-10"], step=2, period=1000, init_cc=0.764,
             init_rmsd=11.24, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,4), dt=0.002)
p97 = show_cc_rmsd(["/home/guest/ScipionUserData/projects/p97/Runs/000671_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/p97/Runs/000733_FlexProtGenesisFit", ],
             length=[1,1,1], labels=["No modes","Mode 7-10"], step=1, period=1000, init_cc=0.764,
             init_rmsd=11.24, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)
p97 = show_cc_rmsd(["/home/guest/ScipionUserData/projects/p97/Runs/001007_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/p97/Runs/001067_FlexProtGenesisFit", ],
             length=[1,1,1], labels=["No modes","Mode 7-10"], step=1, period=1000, init_cc=0.764,
             init_rmsd=11.24, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)

ef2 = show_cc_rmsd(["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003471_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010353_FlexProtGenesisFit",
              ],
             length=[5,5,1,1], labels=["No modes", "Mode 7-12", "2.0", "5.0"], step=5, period=1000, init_cc=0.59,
             init_rmsd=14.2, fvar=1, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002,
                img = "/home/guest/Workspace/Paper_Frontiers/screenschots/EF2.png" )
# chimera /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003471_FlexProtGenesisFit/extra/target.mrc /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003471_FlexProtGenesisFit/extra/run0_.pdb /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010353_FlexProtGenesisFit/extra/run0_.pdb  /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/002703_ProtImportPdb/extra/1n0v_fitted.pdb

ef2_exp = show_cc_rmsd(["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010033_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010276_FlexProtGenesisFit",
              ],
             length=[1,1], labels=["No modes", "Mode 7-12"], step=5, period=1000, init_cc=0.59,
             init_rmsd=14.2, fvar=1, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)

lao = show_cc_rmsd(["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004037_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004126_FlexProtGenesisFit"
              ],
             length=[4,4], labels=["No modes", "Mode 7-12"], step=10, period=100, init_cc=0.82,
             init_rmsd=7.4, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)
#chimera /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004037_FlexProtGenesisFit/extra/target.sit /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004037_FlexProtGenesisFit/extra/run0_.pdb /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004126_FlexProtGenesisFit/extra/run0_.pdb /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003747_ProtImportPdb/extra/1lst.pdb

lao_noise = show_cc_rmsd(["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/013270_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/013330_FlexProtGenesisFit"
              ],
             length=[5,5], labels=["No modes", "Mode 7-12"], step=10, period=100, init_cc=0.82,
             init_rmsd=7.4, fvar=1, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)

malto = show_cc_rmsd(["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004443_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004502_FlexProtGenesisFit"
              ],
             length=[5,5], labels=["No modes", "Mode 7-12"], step=10, period=100, init_cc=0.875,
             init_rmsd=6.5, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)
#chimera /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004443_FlexProtGenesisFit/extra/target.sit /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004443_FlexProtGenesisFit/extra/run0_.pdb /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004502_FlexProtGenesisFit/extra/run0_.pdb /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004188_ProtImportPdb/extra/1anf.pdb

malto_noise = show_cc_rmsd(["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/013707_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/013767_FlexProtGenesisFit"
              ],
             length=[5,5], labels=["No modes", "Mode 7-12"], step=10, period=100, init_cc=0.875,
             init_rmsd=6.5, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)

efg = show_cc_rmsd(["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004894_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005217_FlexProtGenesisFit",
                   # "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005817_FlexProtGenesisFit"
              ],
             length=[5,5,1], labels=["No modes", "Mode 7-12",""], step=10, period=100, init_cc=0.84,
             init_rmsd=8.7, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)
# chimera /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004894_FlexProtGenesisFit/extra/target.sit /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004894_FlexProtGenesisFit/extra/run0_.pdb /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005217_FlexProtGenesisFit/extra/run0_.pdb /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004639_ProtImportPdb/extra/2xex.pdb

efg_noise = show_cc_rmsd(["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/013940_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/014000_FlexProtGenesisFit",
              ],
             length=[5,5], labels=["No modes", "Mode 7-12",""], step=10, period=100, init_cc=0.84,
             init_rmsd=8.7, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)

lacto = show_cc_rmsd(["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005596_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005655_FlexProtGenesisFit",
              ],
             length=[5,5], labels=["No modes", "Mode 7-12",""], step=10, period=100, init_cc=0.88,
             init_rmsd=7.65, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)
# chimera /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005596_FlexProtGenesisFit/extra/target.sit /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005596_FlexProtGenesisFit/extra/run0_.pdb /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005655_FlexProtGenesisFit/extra/run0_.pdb /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005341_ProtImportPdb/extra/1lfg.pdb

lacto_noise = show_cc_rmsd(["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/014191_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/014251_FlexProtGenesisFit",
              ],
             length=[5,5], labels=["No modes", "Mode 7-12",""], step=10, period=100, init_cc=0.88,
             init_rmsd=7.65, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)

nucleosome = show_cc_rmsd(["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010903_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/011000_FlexProtGenesisFit",
              ],
             length=[1,1], labels=["No modes", "Mode 7-12",""], step=10, period=100, init_cc=0.88,
             init_rmsd=0.0, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)

corA = show_cc_rmsd(["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/011448_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/011568_FlexProtGenesisFit",
                   # "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/011696_FlexProtGenesisFit",
              ],
             length=[1,1,1], labels=["No modes", "Mode 7-12",""], step=1, period=500, init_cc=0.71,
             init_rmsd=13.0, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)

corAexp = show_cc_rmsd([
                    "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/011759_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/011819_FlexProtGenesisFit",
                   # "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/011696_FlexProtGenesisFit",
              ],
             length=[1,1,1], labels=["No modes", "Mode 7-12",""], step=50, period=1000, init_cc=0.7,
             init_rmsd=13.2, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)

corA_10_7= show_cc_rmsd([
                    "/home/guest/Workspace/Paper_Frontiers/corA/local",
                   "/home/guest/Workspace/Paper_Frontiers/corA/global",
              ],
             length=[1,1], labels=["No modes", "Mode 7-12"], step=10, period=500, init_cc=0.7,
             init_rmsd=13.2, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)
# chimera /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012293_FlexProtGenesisFit/extra/target.mrc /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012116_FlexProtGenesisFit/extra/run0_.pdb /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012293_FlexProtGenesisFit/extra/run0_.pdb /home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010830_FlexProtGeneratePSF/extra/output.pdb

corAexp7A = show_cc_rmsd([
                    "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012116_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012293_FlexProtGenesisFit",
                   # "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/011696_FlexProtGenesisFit",
              ],
             length=[1,1,1], labels=["No modes", "Mode 7-12",""], step=10, period=5000, init_cc=0.7,
             init_rmsd=13.2, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,4), dt=0.002)

corAexp7A = show_cc_rmsd([
                    "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012359_FlexProtGenesisFit",
                   "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012419_FlexProtGenesisFit",
                   # "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/011696_FlexProtGenesisFit",
              ],
             length=[1,1,1], labels=["No modes", "Mode 7-12",""], step=10, period=500, init_cc=0.7,
             init_rmsd=13.2, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)

corAexpAmber = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001484_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001912_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001972_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/002032_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/002096_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/002156_FlexProtGenesisFit",
              ],
             length=[1,1,1,1,1,1], labels=["local 10.0", "global 10.0","local 15.0", "global 15.0","local 20.0", "global 20.0",], step=25, period=1000, init_cc=0.7,
             init_rmsd=13.2, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)


ak_filter = show_cc_rmsd([
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007321_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007585_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007756_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007933_FlexProtGenesisFit",

              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007381_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007645_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007816_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007993_FlexProtGenesisFit",
              ],
             length=[5,5,5,5,5,5,5,5], labels=["5A", "7A", "9A", "11A","5A", "7A", "9A", "11A"], step=5, period=200, init_cc=0.75,
             init_rmsd=8.12, fvar=1, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002,
                colors = list(pl.cm.Reds(np.linspace(0.5,1,4))) + list(pl.cm.Blues(np.linspace(0.5,1,4))), xlim=(-1,25))
ak_filter.savefig("/home/guest/Documents/VirtualBoxShared/pictures/PaperFrontiers/ak_filter.png", dpi=1000)
ak_noise = show_cc_rmsd([
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008934_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009138_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008478_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008702_FlexProtGenesisFit",

              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008994_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009198_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008538_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008762_FlexProtGenesisFit",
              ],
             length=[5,5,5,5,5,4,5,5], labels=["1","0.5","0.1","0.05","1","0.5","0.1","0.05"],
             step=10, period=100, init_cc=0.75,
             init_rmsd=8.12, fvar=0.01, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002,
             colors = list(pl.cm.Reds(np.linspace(0.5,1,4))) + list(pl.cm.Blues(np.linspace(0.5,1,4))), xlim=(-1,25))
ak_noise.savefig("/home/guest/Documents/VirtualBoxShared/pictures/PaperFrontiers/ak_noise.png", dpi=1000)

fig, ax = plt.subplots(1,1)
protlist = ["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008934_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009138_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008478_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008702_FlexProtGenesisFit",

              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008994_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009198_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008538_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008762_FlexProtGenesisFit",
              ]
rmsd=[]
rmsd_all=[]
for i in range(len(protlist)):
    print("//")
    rmsd = []
    for j in range(4):
        r = np.load(protlist[i] + "/extra/run%i_rmsd.npy" % (j))
        print(r[-1])
        rmsd.append(r[-1])
    rmsd_all.append(rmsd)
ax.plot(np.mean(rmsd_all[:4]))
ax.plot(rmsd_all[4:])

ak_noise2 = show_cc_rmsd([
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008478_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008538_FlexProtGenesisFit",
              ],
             length=[5,5,5,4,5,4,5,5], labels=["MD only", "MD and NMA"],
             step=10, period=100, init_cc=0.75,
             init_rmsd=8.12, fvar=1, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)
ak_MW = show_cc_rmsd([
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012729_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012789_FlexProtGenesisFit",
              ],
             length=[5,5,1], labels=["MD only", "MD and NMA", ""],
             step=10, period=100, init_cc=0.75,
             init_rmsd=8.12, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)
ak_MW = show_cc_rmsd([
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/015722_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/016167_FlexProtGenesisFit",
              ],
             length=[1,1], labels=["MD only", "MD and NMA", ""],
             step=10, period=100,fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002)

# ak.savefig("results/AK_genesis_nma.png", dpi=1000)
# f2.savefig("results/P97_genesis_nma.png", dpi=1000)
# f3.savefig("results/ATPase_genesis_nma.png", dpi=1000)
# ef2.savefig("results/ef2_genesis_nma.png", dpi=1000)
# lao.savefig("results/lao_genesis_nma.png", dpi=1000)
# malto.savefig("results/malto_genesis_nma.png", dpi=1000)
# efg.savefig("results/efg_genesis_nma.png", dpi=1000)
# lacto.savefig("results/lacto_genesis_nma.png", dpi=1000)
# ak_noise.savefig("results/ak_noise_genesis_nma.png", dpi=1000)
# ak_filter.savefig("results/ak_filter_genesis_nma.png", dpi=1000)
# corA.savefig("results/corA_genesis_nma.png", dpi=1000)
# ak_noise2.savefig(fname = "/home/guest/Documents/VirtualBoxShared/pictures/AK_noise2.png", dpi=1000)
# ak_MW.savefig(fname = "/home/guest/Documents/VirtualBoxShared/pictures/AK_MW.png", dpi=1000)

def show_cc_rmsd2(protocol_list, length, labels, period, step, init_cc, init_rmsd, names,
                 fvar=1.0, capthick=10.0, capsize=10.0, elinewidth=1.0, figsize=(10,5), dt=0.002,
                 colors=["tab:blue", "tab:red", "tab:green", "tab:orange",
                         "tab:brown", "tab:olive", "tab:pink", "tab:green", "tab:cyan"],
                 fmts = ["o", "d", "v", "^", "p", "*", "s","x"], img = None, save=None):
    fig, ax = plt.subplots(len(protocol_list)//2, 3, figsize=figsize)
    for i in range(len(protocol_list)):
        if i % 2 : colfmt = 1
        else : colfmt = 0
        n = i//2
        cc = []
        rmsd = []
        for j in range(length[i]):
            cc.append([init_cc[n]] + list(np.load(protocol_list[i]+ "/extra/run"+str(j)+"_cc.npy")))
            rmsd.append([init_rmsd[n]] + list(np.load(protocol_list[i] +"/extra/run"+str(j)+"_rmsd.npy")))
            t = np.arange(len(cc[-1])) * period[n] * dt
        ax[n,0].errorbar(x=t[::step[n]], y=np.mean(cc, axis=0)[::step[n]],  yerr=np.var(cc, axis=0)[::step[n]]*fvar,
                       label=labels[colfmt], color=colors[colfmt], fmt=fmts[colfmt],
                       capthick=capthick, capsize=capsize,elinewidth=elinewidth)
        ax[n,1].errorbar(x=t[::step[n]], y=np.mean(rmsd, axis=0)[::step[n]],yerr=np.var(rmsd, axis=0)[::step[n]]*fvar,
                        color=colors[colfmt], fmt=fmts[colfmt],
                       capthick=capthick, capsize=capsize,elinewidth=elinewidth)
        ax[n,0].plot(t, np.mean(cc, axis=0), "-",color=colors[colfmt])
        ax[n,1].plot(t, np.mean(rmsd, axis=0), "-",color=colors[colfmt])
        ax[n,0].set_ylabel(names[n],  fontsize=15, rotation=0, weight = 'bold')
        if img is not None:
            ax[n,2].imshow(mpimg.imread(img[n]))
            ax[n,2].axis('off')
    ax[0, 0].set_title("CC",  fontsize=15, rotation=0, weight = 'bold')
    ax[0, 1].set_title("RMSD (A)",  fontsize=15, rotation=0, weight = 'bold')
    ax[0, 2].set_title("3D structures",  fontsize=15, rotation=0, weight = 'bold')
    ax[-1, 1].set_xlabel("Simulation Time (ps)")
    ax[-1, 0].set_xlabel("Simulation Time (ps)")
    fig.tight_layout()
    handles, labels = ax[0,0].get_legend_handles_labels()
    handles.append(mpatches.Patch(color='yellow', label='Initial structure'))
    fig.legend(handles = handles, loc='lower right')
    fig.savefig(save, dpi=300)
    return fig

synth = show_cc_rmsd2([
    "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007321_FlexProtGenesisFit",
    "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007381_FlexProtGenesisFit",
    "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004037_FlexProtGenesisFit",
    "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004126_FlexProtGenesisFit",
    "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003471_FlexProtGenesisFit",
    "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010353_FlexProtGenesisFit",
    "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005596_FlexProtGenesisFit",
    "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005655_FlexProtGenesisFit",
],
             length=[5,5, 4,4, 5,5, 5,5], labels=["MD only", "NMA/MD combined"],
        step=[10,10,10,50], period=[100,100,1000,100],
        init_cc=[0.75, 0.82,0.59,0.88], init_rmsd=[8.12, 7.4,14.2,7.65],
        fvar=1, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(15,12), dt=0.002,
    save="/home/guest/Workspace/Paper_Frontiers/screenschots/synth2.png",
    names=  ["Adenylate                  \n Kinase              ",
             "LAO       \n Binding              ",
             "Elongation                   \n factor 2               ",
             "Lactoferrin                    "],
            img = ["/home/guest/Workspace/Paper_Frontiers/screenschots/ak.png",
            "/home/guest/Workspace/Paper_Frontiers/screenschots/LAO.png",
            "/home/guest/Workspace/Paper_Frontiers/screenschots/EF2.png",
            "/home/guest/Workspace/Paper_Frontiers/screenschots/lacto.png",
                   ])

cc, rmsd=  get_cc_rmsd(N=500, prefix="/home/guest/Workspace/Paper_Frontiers/5ftm25ftn/new/run1_"
    , target=Molecule("data/P97/5ftn.pdb"), size=128, voxel_size=1.5, cutoff=6.0, sigma=2.0, step=10, test_idx=True)

mol1 = Molecule("data/corA/3jch.pdb")
mol2 = Molecule("results/run0_.pdb")
chimera_molecule_viewer([mol1,mol2])
mol1.select_chain(["A", "B", "C", "D", "E"])
mol2.select_chain(["A", "B", "C", "D", "E"])
# mol1.save_pdb("data/corA/3jcf_center.pdb")
# mol2.save_pdb("data/corA/3jch_center.pdb")
mol4.center()
mol4.coords += np.mean(mol2.coords, axis=0)

idx =get_mol_conv(mol1, mol2)
get_RMSD_coords(mol1.coords[idx[:,0]], mol2.coords[idx[:,1]])



# mol1.save_pdb("data/lacto/1lfg.pdb")
# mol2.save_pdb("data/lacto/1lfh.pdb")
from src.density import Volume
v = Volume.from_file(file="data/corA/emd_6553.mrc", sigma=2.0, cutoff=10.0)

init = Volume.from_coords(coord=mol1.coords, size=100, sigma=2.0, cutoff=20.0, voxel_size=1.5)
target = Volume.from_coords(coord=mol2.coords, size=100, sigma=2.0, cutoff=20.0, voxel_size=1.5)
# target.show()
# target.save_mrc("data/lacto/1lfh.mrc")
chimera_fit_viewer(mol1, target)
chimera_molecule_viewer([mol1, mol2])
src.density.get_CC(init.data, target.data)
idx = src.functions.get_mol_conv(mol1,mol2)
src.functions.get_RMSD_coords(mol1.coords[idx[0,:]], mol2.coords[idx[1,:]])


from src.molecule import Molecule
import numpy as np

mol = Molecule("data/P97/5ftm.pdb")

new_resNum = np.zeros(mol.n_atoms)
for i in set(mol.chainName):
    chainidx = np.where(mol.chainName == i)[0]
    n=0
    past_idx = 0
    for j in chainidx:
        idx= mol.resNum[j]
        if idx != past_idx:
            n += 1
            past_idx = idx
        new_resNum[j] = n
mol.resNum=np.array(new_resNum, dtype=int)

mol.save_pdb("data/P97/5ftm_smog.pdb")

import mrcfile
from skimage.exposure import match_histograms

with mrcfile.open("data/1AKE/1ake_center.mrc") as mrc:
    mrc_data = np.transpose(np.array(mrc.data), (2,1,0))
with mrcfile.new("data/1AKE/1ake_center2.mrc", overwrite=True) as mrc:
    mrc_data[:10,:,:] = 1000
    mrc.set_data(mrc_data)

map3 = match_histograms(map1, map2)

m1 = Volume.from_file(file="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008994_FlexProtGenesisFit/extra/00001_reconstructed.mrc",
                 sigma=2.0, cutoff=6.0)

m2 = Volume.from_file(file="data/1AKE/1ake_center.mrc",
                 sigma=2.0, cutoff=6.0)

m1.compare_hist(m2)
m1.show()
m1.rescale(method="match",density=m2)
m1.show()
m1.save_mrc(file="data/1AKE/1ake_test.mrc")

t1 = np.load(file="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009360_FlexProtGenesisFit/extra/times.npy")
t3 = np.load(file="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009420_FlexProtGenesisFit/extra/times.npy")
t10 = np.load(file="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009611_FlexProtGenesisFit/extra/times.npy")
t20 = np.load(file="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009674_FlexProtGenesisFit/extra/times.npy")
t50 = np.load(file="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009737_FlexProtGenesisFit/extra/times.npy")
t100 = np.load(file="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009800_FlexProtGenesisFit/extra/times.npy")

print(" Times : ")
print(" \t tlocal = %2.2f +- %2.2f"%(np.mean(t1),np.std(t1)))
print(" \t t3 = %2.2f +- %2.2f"%(np.mean(t3),np.std(t3)))
print(" \t t10 = %2.2f +- %2.2f"%(np.mean(t10),np.std(t10)))
print(" \t t20 = %2.2f +- %2.2f"%(np.mean(t20),np.std(t20)))
print(" \t t50 = %2.2f +- %2.2f"%(np.mean(t50),np.std(t50)))
print(" \t t100 = %2.2f +- %2.2f"%(np.mean(t100),np.std(t100)))

import os
import numpy as np

def compute_rmsd_from_dcd(outputPrefix, targetFname, initFname, N):
    with open("%sdcd2pdb.tcl" % outputPrefix, "w") as f:
        s = ""
        s += "mol load pdb %s dcd %s.dcd\n" % (initFname,outputPrefix)
        s += "set nf [molinfo top get numframes]\n"
        s += "for {set i 0 } {$i < $nf} {incr i} {\n"
        s += "[atomselect top all frame $i] writepdb %stmp$i.pdb\n" % outputPrefix
        s += "}\n"
        s += "exit\n"
        f.write(s)
    os.system("vmd -dispdev text -e %sdcd2pdb.tcl" % outputPrefix)
    from src.molecule import Molecule
    from src.functions import get_mol_conv, get_RMSD_coords
    rmsd = []
    target = Molecule(targetFname)
    mol = Molecule(initFname)
    idx = get_mol_conv(mol, target, ca_only=True)
    if len(idx) > 0:
        rmsd.append(get_RMSD_coords(mol.coords[idx[:, 0]], target.coords[idx[:, 1]]))
        for i in range(N):
            print(i)
            fname = "%stmp%i.pdb" % (outputPrefix, i + 1)
            if os.path.exists(fname):
                mol = Molecule(fname)
                rmsd.append(get_RMSD_coords(mol.coords[idx[:, 0]], target.coords[idx[:, 1]]))
            else:
                print("cant find %s" %fname)
                rmsd.append(0)
    else:
        rmsd = np.zeros(N + 1)
    os.system("rm -f %stmp*" %(outputPrefix))
    np.save(file="%s_rmsd.npy"%outputPrefix, arr=np.array(rmsd))


def read_cc_in_log_file(outputPrefix):
    with open(outputPrefix + ".log", "r") as f:
        header = None
        cc = []
        cc_idx = 0
        for i in f:
            if i.startswith("INFO:"):
                if header is None:
                    header = i.split()
                    for i in range(len(header)):
                        if 'RESTR_CVS001' in header[i]:
                            cc_idx = i
                else:
                    splitline = i.split()
                    if len(splitline) == len(header):
                        cc.append(float(splitline[cc_idx]))
    np.save(file="%s_cc.npy"%outputPrefix, arr=np.array(cc))

def run_molprobity(outputPrefix):
    os.system("~/MolProbity/cmdline/oneline-analysis %s.pdb > %s_molprobity.txt" % (outputPrefix,outputPrefix))
    with open("%s_molprobity.txt"% outputPrefix, "r") as f:
        header = None
        molprob = {}
        for i in f:
            split_line = (i.split(":"))
            if header is None:
                if split_line[0] == "#pdbFileName":
                    header = split_line
            else:
                if len(split_line) == len(header):
                    for i in range(len(header)):
                        molprob[header[i]] = split_line[i]
    np.savetxt(fname="%s_molprobity.txt" % outputPrefix,X=
    np.array([float(molprob["clashscore"]),
              float(molprob["MolProbityScore"]),
              float(molprob["ramaFavored"]) / float(molprob["numRama"]),
              float(molprob["rotaFavored"]) / float(molprob["numRota"])
              ]))


for i in range(0,16):
    outputPrefix = "/home/guest/Workspace/PaperFrontiers/P97/local3/run_r%i"%(i+1)
    compute_rmsd_from_dcd(outputPrefix,
                          targetFname="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000330_FlexProtGeneratePSF/extra/output.pdb",
                          initFname="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000039_FlexProtGeneratePSF/extra/output.pdb",
                          N=140)
    read_cc_in_log_file(outputPrefix)
    # run_molprobity(outputPrefix)


for i in range(0, 16):
    outputPrefix = "/home/guest/Workspace/PaperFrontiers/CorA/51020/local/results20/run_r%i" % (i + 1)
    compute_rmsd_from_dcd(outputPrefix,
                          targetFname="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001871_FlexProtGeneratePSF/extra/output.pdb",
                          initFname="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001374_FlexProtGeneratePSF/extra/output.pdb",
                          N=300)
    read_cc_in_log_file(outputPrefix)
    outputPrefix = "/home/guest/Workspace/PaperFrontiers/CorA/51020/local/results10/run_r%i" % (i + 1)
    compute_rmsd_from_dcd(outputPrefix,
                          targetFname="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001871_FlexProtGeneratePSF/extra/output.pdb",
                          initFname="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001374_FlexProtGeneratePSF/extra/output.pdb",
                          N=300)
    read_cc_in_log_file(outputPrefix)
    outputPrefix = "/home/guest/Workspace/PaperFrontiers/CorA/51020/local/results5/run_r%i" % (i + 1)
    compute_rmsd_from_dcd(outputPrefix,
                          targetFname="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001871_FlexProtGeneratePSF/extra/output.pdb",
                          initFname="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/001374_FlexProtGeneratePSF/extra/output.pdb",
                          N=300)
    read_cc_in_log_file(outputPrefix)
    # run_molprobity(outputPrefix)
    # run_molprobity(outputPrefix)

# LAO
nfit=  [    "008506",    "008580"]
clock_time= [600.4,696.3]
period = 500

#AK
nfit=  [    "000754",    "003692"]
clock_time= [556.1,646.1]
period = 100

#LACTO
nfit=  [    "008748",    "008822"]
clock_time= [7721.176,8224.050]
period = 500

#EF2
nfit=  [    "008900",    "008974"]
clock_time= [23881,24174]
period = 1000

# CORA
nfit=  [    "009425",    "009499"]
clock_time= [27950,30141]
period = 1000

# ABC
nfit=  [    "010519",    "010755"]
clock_time= [26968,28395]
period = 1000

# P97
nfit=  [    "local2",    "global2"]
clock_time= [60340,62020]
period = 100

protlist = ["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008934_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009138_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008478_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008702_FlexProtGenesisFit",

              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008994_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009198_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008538_FlexProtGenesisFit",
              "/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008762_FlexProtGenesisFit",
              ]
/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007162_FlexProtGenesisFit/
percent=0.02
for n in range(len(protlist)) :
    cc = []
    rmsd = []
    # for i in range(16):
    #     cc.append(np.load("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/%s_FlexProtGenesisFit/extra/run_r%i_cc.npy" %(nfit[n],i+1)))
    #     rmsd.append(np.load("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/%s_FlexProtGenesisFit/extra/run_r%i_rmsd.npy"%(nfit[n],i+1)))
    # idx = np.where(rmsd == np.min(rmsd))[0][0]
    # cc = np.array(cc)[idx]
    # rmsd = np.array(rmsd)[idx]

    for i in range(4):
        cc.append(np.load(protlist[n] + "/extra/run%i_cc.npy" % (i)))
        rmsd.append(np.load(protlist[n] + "/extra/run%i_rmsd.npy" % (i)))
    cc = np.mean(cc, axis=0)
    rmsd = np.mean(rmsd, axis=0)

    cc_max= cc.max()
    rmsd_min= rmsd.min()
    cc5p = cc.max() - percent*(cc.max()-cc.min())
    rmsd5p = rmsd.min() + percent*(rmsd.max() - rmsd.min())
    cctime = np.min(np.where(cc>=cc5p)[0])*period*0.002
    rmsdtime = np.min(np.where(rmsd<=rmsd5p)[0])*period*0.002
    rmsdclocktime =0#(rmsdtime /((len(rmsd)-1)*period*0.002)) * clock_time[n]
    print("ccmax %.3f cctime %.1f rmsdmin %.3f rmsdtime %.1f rmsdclocktime %.1f" %(cc_max, cctime, rmsd_min, rmsdtime,rmsdclocktime) )

from src.molecule import Molecule
from src.functions import get_mol_conv,compute_pca
target = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008330_ProtImportPdb/extra/1n0u_target.pdb")
init = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006348_FlexProtGeneratePSF/extra/output.pdb")
idx = get_mol_conv(init, target)

data = []
for i in range(16):
    mol = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008900_FlexProtGenesisFit/extra/run_r%i.pdb" %(i+1))
    data.append(mol.coords[idx[:,0]].flatten())
for i in range(13):
    mol = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008974_FlexProtGenesisFit/extra/run_r%i.pdb" %(i+1))
    data.append(mol.coords[idx[:,0]].flatten())

data.append(init.coords[idx[:,0]].flatten())
data.append(target.coords[idx[:,1]].flatten())

ef2 = compute_pca(data, length=[16,13,1,1], labels=["MD", "NMMD", "Init.", "Target"], save=None, n_components=2, figsize=(4,3), lim=[0.157,0.208], lim2=[-0.9,0.3])
ef2.savefig("/home/guest/Documents/VirtualBoxShared/pictures/PaperFrontiers/EF2_pca.png", dpi=1000)

target = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010487_FlexProtGeneratePSF/extra/output.pdb")
init = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010295_FlexProtGeneratePSF/extra/output.pdb")
idx = get_mol_conv(init, target)
data = []
for i in range(16):
    mol = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010519_FlexProtGenesisFit/extra/run_r%i.pdb" %(i+1))
    data.append(mol.coords[idx[:,0]].flatten())
for i in range(16):
    mol = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010755_FlexProtGenesisFit/extra/run_r%i.pdb" %(i+1))
    data.append(mol.coords[idx[:,0]].flatten())

data.append(init.coords[idx[:,0]].flatten())
data.append(target.coords[idx[:,1]].flatten())
abc = compute_pca(data, length=[16,16,1,1], labels=["MD", "NMMD",  "Init.", "Target"], save=None, n_components=2, figsize=(4,3), lim=[-0.1726,-0.1696], lim2=[-0.95,0.27])
abc.savefig("/home/guest/Documents/VirtualBoxShared/pictures/PaperFrontiers/ABC_pca.png", dpi=1000)

target = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000330_FlexProtGeneratePSF/extra/output.pdb")
init = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000039_FlexProtGeneratePSF/extra/output.pdb")
idx = get_mol_conv(init, target)
data = []
for i in range(16):
    mol = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/P97/local2/run_r%i.pdb" %(i+1))
    data.append(mol.coords[idx[:,0]].flatten())
for i in range(16):
    mol = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/P97/global2/run_r%i.pdb" %(i+1))
    data.append(mol.coords[idx[:,0]].flatten())

data.append(init.coords[idx[:,0]].flatten())
data.append(target.coords[idx[:,1]].flatten())
p97 = compute_pca(data, length=[16,16,1,1], labels=["MD", "NMMD",  "Init.", "Target"], save=None, n_components=2, figsize=(4,3), lim=[-0.1772,-0.163], lim2=[-0.61,0.4])
p97.savefig("/home/guest/Documents/VirtualBoxShared/pictures/PaperFrontiers/p97_pca.png", dpi=1000)

target = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008293_ProtImportPdb/extra/1lfh_target.pdb")
init = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004537_FlexProtGeneratePSF/extra/output.pdb")
idx = get_mol_conv(init, target)
data = []
for i in range(16):
    mol = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008748_FlexProtGenesisFit/extra/run_r%i.pdb" %(i+1))
    data.append(mol.coords[idx[:,0]].flatten())
for i in range(16):
    mol = Molecule("/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/008822_FlexProtGenesisFit/extra/run_r%i.pdb" %(i+1))
    data.append(mol.coords[idx[:,0]].flatten())

data.append(init.coords[idx[:,0]].flatten())
data.append(target.coords[idx[:,1]].flatten())
lacto = compute_pca(data, length=[16,16,1,1], labels=["MD", "NMMD",  "Init.", "Target"], save=None, n_components=2, figsize=(4,3),lim2=[-0.42,0.79])
lacto.savefig("/home/guest/Documents/VirtualBoxShared/pictures/PaperFrontiers/lacto_pca.png", dpi=1000)


from src.molecule import Molecule
from src.density import Volume
from joblib import load

init = Molecule("/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000530_FlexProtGeneratePSF/extra/output.pdb")
fnModes = np.array(["/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000567_FlexProtNMA/modes/vec."+str(i+7) for i in range(13)])
init.set_normalModeVec(fnModes)
optFlow = load("/home/guest/Downloads/Optical_flow_test_with_Remi/optical_flow2.pkl")
optFlow = np.transpose(optFlow, (0,3, 2, 1))
optFlowAtom = np.zeros((init.n_atoms,3))

target = init.copy()
voxel_size= 1.5
origin = -np.ones(3) * 100//2

for i in range(init.n_atoms):
    coord = init.coords[i]/voxel_size - origin
    floor = np.ndarray.astype(np.floor(coord), dtype=int)
    comb = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],
                     [1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    optFlows = []
    for j in floor+comb:
        if any(j < 0) or any(j >= optFlow.shape[1:]):
            print("error :%s" % j)
        optFlows.append(optFlow[:, j[0],j[1],j[2]])
    optFlowAtom[i] = np.mean(optFlows, axis=0)
    print("Optical flow for atom %i : %s"%(i,str(optFlowAtom[i])))
    target.coords[i] = init.coords[i] + -optFlowAtom[i]
target.save_pdb("test_optFlow.pdb")

# for i in range(init.normalModeVec.shape[1]):
#     m = np.dot(optFlowAtom.flatten(), init.normalModeVec[:,i,:].flatten())
#     print("Modes #%i = %f"%(i+7, m))


from src.molecule import Molecule
pdb = Molecule("data/Ribosome/3j77_cat2.pdb")







##################################################################################
#  ABC HETERO SPACE
#################################################################################
from src.molecule import Molecule
from src.functions import compute_pca,get_mols_conv
from src.io import create_psf
import os
from src.density import Volume
import mrcfile
import numpy as np

n_mols= 9
voxel_size= 1.077
labels = ["6raf","6rag","6rah","6rai","6raj","6rak","6ral","6ram","6ran"]
emlabels = ["4773","4774","4775","4776","4777","4778","4779","4780","4781"]

mols = []
mols.append(Molecule("data/ABC/6raf.pdb"))
mols.append(Molecule("data/ABC/6rag.pdb"))
mols.append(Molecule("data/ABC/6rah.pdb"))
mols.append(Molecule("data/ABC/6rai.pdb"))
mols.append(Molecule("data/ABC/6raj.pdb"))
mols.append(Molecule("data/ABC/6rak.pdb"))
mols.append(Molecule("data/ABC/6ral.pdb"))
mols.append(Molecule("data/ABC/6ram.pdb"))
mols.append(Molecule("data/ABC/6ran.pdb"))

mol_ref = mols[0].copy()
mol_ref.center()
vol_ref = Volume.from_coords(mol_ref.coords, size=256, sigma=2.0, voxel_size=1.077, cutoff=10.0)

for i in range(n_mols):

    print("\n> Copying data/ABC/emd_%s.map to data/ABC/emd_%s.mrc"% (emlabels[i],emlabels[i]))
    os.system("cp data/ABC/emd_%s.map data/ABC/emd_%s.mrc"% (emlabels[i],emlabels[i]))


    print("\n> Filtering data/ABC/emd_%s.mrc to data/ABC/emd_%s_filter.mrc" % (emlabels[i], emlabels[i]))
    cmd = "xmipp_transform_filter -i data/ABC/emd_%s.mrc:mrc -o data/ABC/emd_%s_filter.mrc --fourier low_pass 0.215400 0.010770"% (emlabels[i],emlabels[i])
    os.system(cmd)

    print("\n> Filtering data/ABC/emd_%s.mrc to data/ABC/emd_%s_filter.mrc" % (emlabels[i], emlabels[i]))
    with mrcfile.open("data/ABC/emd_%s_filter.mrc" %(emlabels[i])) as mrc:
        mrc_data = mrc.data

    print("\n> Set origin data/ABC/emd_%s_filter2.mrc"% (emlabels[i]))
    fnMRC = "data/ABC/emd_%s_filter2.mrc" % (emlabels[i])
    with mrcfile.new(fnMRC, overwrite=True) as mrc:
        mrc.set_data(mrc_data)
        mrc.voxel_size = voxel_size
        origin = -voxel_size * (np.array(mrc_data.shape)) / 2
        mrc.header['origin']['x'] = origin[0]
        mrc.header['origin']['y'] = origin[1]
        mrc.header['origin']['z'] = origin[2]
        mrc.update_header_from_data()
        mrc.update_header_stats()

    print("\n> Histogram matching data/ABC/emd_%s_match.mrc" % (emlabels[i]))
    fnvol = "data/ABC/emd_%s_match.mrc"%emlabels[i]
    vol = Volume.from_file("data/ABC/emd_%s_filter2.mrc"%emlabels[i])
    vol.voxel_size=voxel_size
    print("\t -> EM map %s of size %ix%ix%i and voxel size %f "% (emlabels[i], vol.data.shape[0], vol.data.shape[1], vol.data.shape[2], vol.voxel_size))
    vol.rescale(density = vol_ref, method="match")
    vol.save_mrc(fnvol, origin=0.0)
    with open("data/ABC/tmp.sh", "w") as f:
        f.write("#!/bin/bash \n")
        f.write("~/Situs/bin/map2map data/ABC/emd_%s_match.mrc data/ABC/emd_%s.sit <<< \'1\' \n"% (emlabels[i],emlabels[i]))
        f.write("exit")
    os.system("/bin/bash data/ABC/tmp.sh")
    os.system("rm -f data/ABC/tmp.sh")

for i in range(n_mols):
    with open("data/ABC/INP_min_%s"%labels[i], "w") as f:
        f.write("[INPUT]\n")
        f.write("topfile = top_all36_prot.rtf\n")
        f.write("parfile = par_all36_prot.prm\n")
        f.write("pdbfile = %s_PSF.pdb\n" %labels[i])
        f.write("psffile = %s.psf\n" %labels[i])
        f.write("[OUTPUT]\n")
        f.write("dcdfile = %s_min.dcd\n" %labels[i])
        f.write("rstfile = %s_min.rst\n" %labels[i])
        f.write("[ENERGY]\n")
        f.write("forcefield = CHARMM  # CHARMM force field\n")
        f.write("electrostatic = CUTOFF  # use cutoff scheme for non-bonded terms\n")
        f.write("switchdist = 23.0  # switch distance\n")
        f.write("cutoffdist = 25.0  # cutoff distance\n")
        f.write("pairlistdist = 27.0  # pair-list distance\n")
        f.write("implicit_solvent = NONE  # use GBSA implicit solvent model\n")
        f.write("vdw_force_switch = YES\n")
        f.write("[MINIMIZE]\n")
        f.write("method = SD  # Steepest descent\n")
        f.write("nsteps = 100\n")
        f.write("eneout_period = 10  # energy output period\n")
        f.write("crdout_period = 100\n")
        f.write("rstout_period = 100\n")
        f.write("nbupdate_period = 10  # nonbond update period\n")
        f.write("[BOUNDARY]\n")
        f.write("type = NOBC  # No periodic boundary condition")
for i in range(n_mols):
    os.system("~/genesis_nma/bin/atdyn data/ABC/INP_min_%s"%labels[i])


for i in range(n_mols):
    for j in range(n_mols):
        print("<NMMD_ID> %i_%s_%s"%(i*n_mols + j, labels[i], emlabels[j]))
        with open("data/ABC/INP_fit_%s_%s"%(labels[i], emlabels[j]), "w") as f:
            f.write("[INPUT]\n")
            f.write("topfile = top_all36_prot.rtf\n")
            f.write("parfile = par_all36_prot.prm\n")
            f.write("pdbfile = %s_PSF.pdb\n" %labels[i])
            f.write("psffile = %s.psf\n" %labels[i])
            f.write("rstfile = %s_min.rst\n" %labels[i])
            f.write("[OUTPUT]\n")
            f.write("dcdfile = run_%s_%s.dcd\n"%(labels[i], emlabels[j]))
            f.write("rstfile = run_%s_%s.rst\n"%(labels[i], emlabels[j]))
            f.write("pdbfile = run_%s_%s.pdb\n"%(labels[i], emlabels[j]))
            f.write("[ENERGY]\n")
            f.write("forcefield = CHARMM  # CHARMM force field\n")
            f.write("electrostatic = CUTOFF  # use cutoff scheme for non-bonded terms\n")
            f.write("switchdist   = 5.0\n")
            f.write("cutoffdist   = 7.0\n")
            f.write("pairlistdist = 10.0\n")
            f.write("vdw_force_switch = YES\n")
            f.write("implicit_solvent = NONE\n")
            f.write("[DYNAMICS]\n")
            f.write("integrator = VVER\n")
            f.write("nsteps = 25000 # = 1 ns\n")
            f.write("timestep = 0.002  # ps = 2 femtosecond\n")
            f.write("eneout_period = 100\n")
            f.write("crdout_period = 100\n")
            f.write("rstout_period = 25000\n")
            f.write("nbupdate_period = 10\n")
            f.write("[CONSTRAINTS]\n")
            f.write("rigid_bond = NO  # use SHAKE\n")
            f.write("[ENSEMBLE]\n")
            f.write("ensemble = NVT  # constant temperature\n")
            f.write("tpcontrol = LANGEVIN\n")
            f.write("temperature = 300.0 #K\n")
            f.write("[BOUNDARY]\n")
            f.write("type = NOBC  # No periodic boundary condition\n")
            f.write("[SELECTION]\n")
            f.write("group1 = all and not hydrogen\n")
            f.write("[RESTRAINTS]\n")
            f.write("nfunctions = 1\n")
            f.write("function1 = EM  # apply restraints from EM density map\n")
            f.write("constant1 = 40000\n")
            f.write("select_index1 = 1  # apply restraint force on protein heavy atoms\n")
            f.write("[EXPERIMENTS]\n")
            f.write("emfit = YES  # perform EM flexible fitting\n")
            f.write("emfit_target = emd_%s.sit\n" %emlabels[j])
            f.write("emfit_sigma = 2.0\n")
            f.write("emfit_tolerance = 0.01\n")
            f.write("emfit_period = 1  # emfit force update period\n")

##################################################################################
#  ABC HETERO SPACE tmp
#################################################################################
from src.molecule import Molecule
from src.functions import compute_pca,get_mols_conv
from src.io import create_psf
import os
from src.density import Volume
import mrcfile
import numpy as np

n_mols= 9
voxel_size= 1.077
labels = ["6raf","6rag","6rah","6rai","6raj","6rak","6ral","6ram","6ran"]
emlabels = ["4773","4774","4775","4776","4777","4778","4779","4780","4781"]

# DCD to PDB
for i in range(n_mols):
    for j in range(n_mols):
        if not os.path.exists("data/ABC/tmp/run_%s_%stmp0.pdb"%(labels[i], emlabels[j])):
            if os.path.exists("data/ABC/run_%s_%s.pdb"%(labels[i], emlabels[j])):
                print("data/ABC/tmp/run_%s_%stmp$i.pdb"%(labels[i], emlabels[j]))
                with open("dcd2pdb.tcl", "w") as f:
                    s = ""
                    s += "mol load pdb data/ABC/run_%s_%s.pdb dcd data/ABC/run_%s_%s.dcd\n"%(labels[i], emlabels[j],labels[i], emlabels[j])
                    s += "set nf [molinfo top get numframes]\n"
                    s += "for {set i 0 } {$i < 200} {incr i 1} {\n"
                    s += "[atomselect top all frame $i] writepdb data/ABC/tmp/run_%s_%stmp$i.pdb\n"%(labels[i], emlabels[j])
                    s += "}\n"
                    s += "exit\n"
                    f.write(s)
                os.system("vmd -dispdev text -e dcd2pdb.tcl")




##################################################################################
#  ABC HETERO SPACE results
#################################################################################
from src.molecule import Molecule
from src.functions import compute_pca,get_mols_conv
from src.io import create_psf
import os
from src.density import Volume
import mrcfile
import numpy as np

n_mols= 9
voxel_size= 1.077
labels = ["6raf","6rag","6rah","6rai","6raj","6rak","6ral","6ram","6ran"]
emlabels = ["4773","4774","4775","4776","4777","4778","4779","4780","4781"]

mols = []
mols.append(Molecule("data/ABC/6raf_PSF.pdb"))
mols.append(Molecule("data/ABC/6rag_PSF.pdb"))
mols.append(Molecule("data/ABC/6rah_PSF.pdb"))
mols.append(Molecule("data/ABC/6rai_PSF.pdb"))
mols.append(Molecule("data/ABC/6raj_PSF.pdb"))
mols.append(Molecule("data/ABC/6rak_PSF.pdb"))
mols.append(Molecule("data/ABC/6ral_PSF.pdb"))
mols.append(Molecule("data/ABC/6ram_PSF.pdb"))
mols.append(Molecule("data/ABC/6ran_PSF.pdb"))
for i in mols:
    i.allatoms2carbonalpha()

idx=  get_mols_conv(mols)

colors = ["tab:red", "tab:blue", "tab:orange", "tab:green",
          "tab:brown", "tab:olive", "tab:pink", "tab:purple", "tab:cyan", "tab:gray"]
data_pca=[]
mol_res = []
length=[]
labels_pca= []
alphas = []
colors_pca=[]
marker=[]
traj=[]
n_mols_target=8
n_mols_fit=8
len_traj=200
for i in range(n_mols_fit):
    # TRAJ
    for j in range(n_mols_target):
        if j == 0:
            length.append(0)
            labels_pca.append(emlabels[i] + "_traj")
            alphas.append(0.5)
            colors_pca.append(colors[i])
            marker.append('-')
            traj.append(n_mols_target)
        for k in range(1,len_traj-1):
            tmp_file = "data/ABC/tmp/run_%s_%stmp%i.npy" % (labels[j], emlabels[i], k)
            if os.path.exists(tmp_file):
                tmp_arr = np.load(file=tmp_file)
            else:
                m = Molecule("data/ABC/tmp/run_%s_%stmp%i.pdb" % (labels[j], emlabels[i], k ))
                m.allatoms2carbonalpha()
                tmp_arr = m.coords[idx[:, j]].flatten()
                np.save(file="data/ABC/tmp/run_%s_%stmp%i.npy" % (labels[j], emlabels[i], k ), arr=tmp_arr)
            data_pca.append(tmp_arr)
            length[-1] += 1
for i in range(n_mols_fit):
    # Fit
    for j in range(n_mols_target):
        if j ==0:
            length.append(0)
            labels_pca.append(emlabels[i] + "_fit")
            alphas.append(1)
            colors_pca.append(colors[i])
            marker.append('^')
            traj.append(1)
        npy_file="data/ABC/run_%s_%s.npy"%(labels[j], emlabels[i])
        if os.path.exists(npy_file):
            tmp_arr = np.load(file=npy_file)
        else:
            m = Molecule("data/ABC/run_%s_%s.pdb"%(labels[j], emlabels[i]))
            m.allatoms2carbonalpha()
            tmp_arr=m.coords[idx[:, j]].flatten()
            np.save(file=npy_file, arr=tmp_arr)
        data_pca.append(tmp_arr)
        length[-1] += 1

    # Target
    data_pca.append(mols[i].coords[idx[:, i]].flatten())
    length.append(1)
    labels_pca.append(labels[i])
    alphas.append(1)
    colors_pca.append(colors[i])
    marker.append('o')
    traj.append(1)

fig, ax =compute_pca(data=data_pca, length=length, labels=labels_pca, n_components=3,
                    figsize=(7,5), alphas=alphas, colors=colors_pca, marker=marker,
                    traj=traj, legend=False)
fig.show()
ax.set_facecolor("white")
fig.savefig("/home/guest/Pictures/ABC_heterospace_2D3.png", dpi=1000)

import numpy as np
import matplotlib.pyplot as plt

with open("/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012803_FlexProtGenesisFit/extra/run_r1.nmmdglobalpos", "r") as f:
    globalpos = np.array( list(np.loadtxt(f)))
with open("/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012803_FlexProtGenesisFit/extra/run_r1.nmmdlocalpos", "r") as f:
    localpos = np.array(list(np.loadtxt(f)))
with open("/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012874_FlexProtGenesisFit/extra/run_r1.nmmdlocalpos", "r") as f:
    localpos2 = np.array(list(np.loadtxt(f)))


plt.plot(globalpos[1:]-globalpos[:-1])
plt.plot(localpos[1:]-localpos[:-1])
plt.plot(localpos2[1:]-localpos2[:-1])
plt.axhline(0)



import numpy as np
import os

n_mols= 9
labels = ["6raf","6rag","6rah","6rai","6raj","6rak","6ral","6ram","6ran"]
emlabels = ["4773","4774","4775","4776","4777","4778","4779","4780","4781"]
for i in labels:
    for j in emlabels:
        for k in range(0,200,10):
            f= "run_%s_%stmp%i.pdb" %(i,j,k)
            os.system("~/MolProbity/cmdline/oneline-analysis %s > %s_molprobity.txt" % (f,f))
            with open("%s_molprobity.txt"% f, "r") as mpf:
                header = None
                molprob = {}
                for i in mpf:
                    split_line = (i.split(":"))
                    if header is None:
                        if split_line[0] == "#pdbFileName":
                            header = split_line
                    else:
                        if len(split_line) == len(header):
                            for i in range(len(header)):
                                molprob[header[i]] = split_line[i]
            with open("%s_molprobity.txt"% f, "w") as mpf:
                mpf.write(molprob["MolProbityScore"])





##############################################################################################
#Clock time :
import re
n = [
#LAO
    "003997",
    "004430",
# AK
    "007162",
    "012724",
#LACTO
    "004647",
    "005083",
#EF2
    "006459",
    "013591",
#ABC
    "010519",
    "010755",
#P97
    "000306",
    "000376"]

percent=0.01
period=1000
totallocal = []
totalglobal = []
averlocal = []
averglobal = []
meanlocal = []
meanglobal = []
minlocal = []
minglobal = []

ratio = []

for nloop in range(len(n)):
    cc = []
    rmsd = []
    print(nloop)
    sfx1 ="/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/"
    sfx2 ="_FlexProtGenesisFit/extra/run"
    if not os.path.exists("%s%s_FlexProtGenesisFit/logs/run.stdout"%(sfx1,n[nloop])) :
        sfx1 = "/home/guest/ScipionUserData/projects/Remi_P97_paper2021/Runs/"
    with open("%s%s_FlexProtGenesisFit/logs/run.stdout"%(sfx1,n[nloop])) as f:
        for line in f:
            if line.startswith("  total time      ="):
                clock_time = (float(re.findall("\d+\.\d+",line)[0]))

    for i in range(16):
        if not os.path.exists("%s%s%s_r1.log" % (sfx1, n[nloop], sfx2)):
            sfx = "%s%s%s1_remd"% (sfx1, n[nloop], sfx2)
        else:
            sfx= "%s%s%s_r"% (sfx1, n[nloop], sfx2)
        cc.append(np.load("%s%i_cc.npy" % (sfx,i+1)))
        rmsd.append(np.load("%s%i_rmsd.npy" % (sfx,i+1)))

    # CC mean
    cc_mean = np.mean(cc, axis=0)
    cc_max= cc_mean.max()
    cc5p = cc_mean.max() - percent*(cc_mean.max()-cc_mean.min())
    cctime = np.min(np.where(cc_mean>=cc5p)[0])*period*0.002

    #RMSD Aver
    rmsd_mean = np.mean(rmsd, axis=0)
    rmsd_min= rmsd_mean.min()
    rmsd5p = rmsd_mean.min() + percent*(rmsd_mean.max() - rmsd_mean.min())
    rmsdtime = np.min(np.where(rmsd_mean<=rmsd5p)[0])*period*0.002
    rmsdclocktime = (rmsdtime /((len(rmsd_mean)-1)*period*0.002)) #"* clock_time

    #RMSD Min
    idx = np.where(rmsd == np.min(rmsd))[0][0]

    #RMSD Mean
    rmsdclocktimeMean = []
    for i in range(16):
        rmsd5pMean = rmsd[i].min() + percent * (rmsd[i].max() - rmsd[i].min())
        rmsdtimeMean = np.min(np.where(rmsd[i] <= rmsd5pMean)[0]) * period * 0.002
        rmsdclocktimeMean.append((rmsdtimeMean / ((len(rmsd[i]) - 1) * period * 0.002))) #* clock_time
        if i == idx :
            rmsdclocktimeMin = (rmsdtimeMean / ((len(rmsd[i]) - 1) * period * 0.002)) #* clock_time
    rmsdclocktimeMean = np.mean(rmsdclocktimeMean)


    if nloop % 2 == 0:
        totallocal.append(clock_time)
        averlocal.append(rmsdclocktime)
        meanlocal.append(rmsdclocktimeMean)
        minlocal.append(rmsdclocktimeMin)
    else:
        totalglobal.append(clock_time)
        averglobal.append(rmsdclocktime)
        meanglobal.append(rmsdclocktimeMean)
        minglobal.append(rmsdclocktimeMin)

totallocal = np.array(totallocal)
totalglobal = np.array(totalglobal)
averlocal = np.array(averlocal)
averglobal = np.array(averglobal)
meanlocal = np.array(meanlocal)
meanglobal = np.array(meanglobal)
minlocal = np.array(minlocal)
minglobal = np.array(minglobal)

print("Speed increase " )
print("* aver  %.3f" %np.mean(1 -(averglobal/averlocal)))
print(1 -(averglobal/averlocal))
print("* mean  %.3f" %np.mean(1 -(meanglobal/meanlocal)))
print(1 -(meanglobal/meanlocal))
print("* min  %.3f" %np.mean(1 -(minglobal/minlocal)))
print(1 -(minglobal/minlocal))




# print("ccmax %.3f cctime %.1f rmsdmin %.3f rmsdtime %.1f rmsdclocktime %.1f" %(np.max(cc), cctime, np.min(rmsd), rmsdtime,rmsdclocktime) )
# print("TOTAL Time --- %.3f" %clock_time)
# print("* ccmax  %.3f" %np.max(cc))
# print("* rmsdmin  %.3f" %np.min(rmsd))
# print("* clocktime  %.3f" %(clock_time))
# print("* rmsdtime  %.3f" %(rmsdclocktime))
# print("* rmsdtime  %.3f" %(rmsdclocktimeMean))
# print("* rmsdtime  %.3f" %(rmsdclocktimeMin))
#
