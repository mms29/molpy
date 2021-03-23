import time

import numpy as np

import src.io
from src.forcefield import *
from src.functions import *

init =src.io.read_pdb("data/AK_tomos/AK.pdb")
init.add_modes("data/AK_tomos/modes/vec.", n_modes=4)
init.select_atoms(pattern='CA')
init.set_forcefield()


x = np.zeros(3)
q = np.zeros(init.modes.shape[1])
angles = np.random.uniform(-0.5,0.5,3)


R = generate_euler_matrix(angles = angles)

t = time.time()
dangles1 =get_euler_autograd(coord = init.coords[0], A = init.modes[0], x=x, q=q, angles=angles)
print(time.time() - t)

t = time.time()
dangles2 =get_euler_grad(angles=angles, coord = init.coords[0])
print(time.time() - t)

print(dangles1[2] - np.sum(dangles2, axis=1))

d = {"x": np.array([0.0,0.0,0.0]), "angles": np.array([0.0,0.1,0.2])}
a = get_autograd(init, d)

########################################################################################################################

import src.simulation
from src.viewers import chimera_structure_viewer
from src.flexible_fitting import *
from src.molecule import Molecule
import copy


init =Molecule.from_file("data/AK/AK_PSF.pdb")
init.center_structure()
init.add_modes("data/AK/modes_psf/vec.", n_modes=4)
init.select_atoms(pattern='CA')
init.set_forcefield()

 # q = [300,-100,0,0]
q = [0,0,0,0]
# angles= [-0.1,0.2, 0.5]
target = src.simulation.nma_deform(init, q)
# target.rotate(angles)
shift = [8.3,5.2,7.0]
target.coords += shift

size=64
voxel_size=2.2
threshold= 4
gaussian_sigma=2
target_density = Image.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
init_density = Image.from_coords(coord=init.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
# init_density.show()
# target_density.show()

params ={
    "lb" : 100,
    "lp" : 1,
    "max_iter": 10,
    "criterion" :False,

    "q_init" : np.zeros(4),
    "q_dt" : 0.1,
    "q_mass" : 1,

    "angles_init" : np.zeros(3),
    "angles_dt" : 0.0001,
    "angles_mass" : 1,
    "langles" :0,

    "shift_init": np.zeros(3),
    "shift_dt": 0.001,
    "shift_mass": 1,
    "lshift": 0,
}
n_iter=20
n_warmup = n_iter // 2

fit  =FlexibleFitting(init = init, target= target_density, vars=["shift"], params=params,
                      n_iter=n_iter, n_warmup=n_warmup, n_chain=1, verbose=2)

fit.HMC()
fit.show()
fit.show_3D()
chimera_structure_viewer([fit.res["mol"], target, init])


test = copy.deepcopy(init)
test.rotate(fit.res["angles"])
chimera_structure_viewer([test, target])




############################################################
###  AUTOGRAD
############################################################





import autograd.numpy as npg
from autograd import elementwise_grad

def get_energy_autograd(mol, params):
    coord = npg.array(mol.coords)
    if "x" in params:
        coord+= params["x"]
    if "q" in params:
        coord+= npg.dot(params["q"], mol.modes)
    if "angles" in params:
        coord = npg.dot(src.functions.generate_euler_matrix(params["angles"]), coord.T).T

    U_bonds = get_energy_bonds(coord, mol.bonds, mol.prm)
    U_angles = get_energy_angles(coord, mol.angles, mol.prm)
    U_dihedrals = get_energy_dihedrals(coord, mol.dihedrals, mol.prm)

    return U_bonds + U_angles + U_dihedrals

def get_autograd(mol, params):
    grad = elementwise_grad(get_energy_autograd, 1)
    return grad(mol, params)
d = {"x":np.array([0.0,0.0,0.0]),
     "q": np.array([0.0,0.0,0.0,0.0]),
     "angles" : np.array([0.0,0.0,0.0])}
get_energy_autograd(init,  d)
a = get_autograd(init,  d)

from src.forcefield import get_gradient_auto
get_gradient_auto(init, {"x": init.coords, "angles": [0.0,.0,.0]})


############################################################
###  FIND CHAIN IN PSEUDOATOMS
############################################################

from src.molecule import Molecule
from src.viewers import chimera_molecule_viewer
import numpy as np
import matplotlib.pyplot as plt
import copy

mol =Molecule.from_file(file="/home/guest/ScipionUserData/projects/BayesianFlexibleFitting/Runs/001192_FlexProtConvertToPseudoAtoms/pseudoatoms.pdb")

dist = np.zeros((mol.n_atoms, mol.n_atoms))
for i in range(mol.n_atoms):
    for j in range(mol.n_atoms):
        dist[i,j] = np.linalg.norm(mol.coords[i]-mol.coords[j])
        if i==j :
            dist[i,j]=100.0
plt.pcolormesh(dist)
np.where(dist==dist.min())

mol.set_forcefield()
U =mol.get_energy()

n=0
l=[]
while(n<1000):
    for i in range(mol.n_atoms-1):
        molc = copy.deepcopy(mol)
        tmp1 = copy.deepcopy(molc.coords[i])
        tmp2 = copy.deepcopy(molc.coords[i+1])
        molc.coords[i] = tmp2
        molc.coords[i+1] = tmp1
        molc.set_forcefield()
        Uc = molc.get_energy()
        if Uc < U:
            print("yes"+str(n)+" ; "+str(np.mean(l)))
            mol = molc
            U=Uc
            l.append(1)

        else :
            l.append(0)
        if len(l)>20:
            l=l[1:]
    n+=1



############################################################
###  P97
############################################################

from src.molecule import Molecule
import numpy as np
import copy
from src.viewers import chimera_molecule_viewer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

init = Molecule.from_file("data/P97/5ftm.pdb")
init.center_structure()
fnModes = np.array(["data/P97/modes_atoms/vec."+str(i+7) for i in range(4)])
init.add_modes(fnModes)

N=1000
mols = []
q = np.random.randint(0,2,(N,6))
for n in range(N):
    mol = copy.deepcopy(init)
    for i in range(mol.n_chain):
        mol.coords[mol.chain_id[i]:mol.chain_id[i+1]] += np.dot(np.array([0,0,q[n,i]*-1500,0]), mol.modes[mol.chain_id[i]:mol.chain_id[i+1]])
    mols.append(mol)

# chimera_molecule_viewer([mols[1]])


n_components=mol.n_chain
res_arr = np.array([i.coords.flatten() for i in mols])
res_pca = PCA(n_components=n_components)
res_pca.fit(res_arr.T)

i=0
plt.figure()
plt.plot(res_pca.components_[0+i], res_pca.components_[1+i], 'o',label='ground truth',markeredgecolor='black')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(res_pca.components_[0], res_pca.components_[1], res_pca.components_[2], s=10, label='ground truth')


############################################################
###  PRODY
############################################################

from src.molecule import Molecule
import numpy as np
import copy
from src.viewers import chimera_molecule_viewer,chimera_fit_viewer
from src.constants import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.functions import compute_pca
from src.density import Volume
from src.flexible_fitting import FlexibleFitting, multiple_fitting

init = Molecule.from_file("data/AK/AK_prody/AK.pdb")
init.center_structure()
fnModes = np.array(["/home/guest/ScipionUserData/projects/nma_calculator/Runs/000759_FlexProtNMA/modes/vec."+str(i+7) for i in range(6)])
init.add_modes(fnModes)
init.select_atoms(pattern="CA")
init.set_forcefield()
init_density = Volume.from_coords(coord=init.coords, voxel_size=1.5, size=64, threshold=5, sigma=2)

N=41
mols=[]
targets = []
for i in range(N):
    mol = Molecule.from_file("/home/guest/Downloads/comd_tutorial_files/frame_reduced_"+str(i)+".pdb")
    mol.select_atoms()
    mols.append(mol)
    target = Volume.from_coords(coord=mol.coords, voxel_size=1.5, size=64, threshold=5, sigma=2)
    targets.append(target)

params ={
    "biasing_factor" : 100,
    "potential_factor" : 1,

    "local_dt" : 0.01,
    "global_dt" : 0.1,
    "shift_dt" : 0.0001,

    "n_iter":20,
    "n_warmup":10,
    "n_step": 20,
    "criterion": False,

    "local_mass" :  1
}

n_chain = 4
n_proc = 12
models_global=[]
models_local =[]
models_glocal=[]
for i in targets:
    models_global.append(FlexibleFitting(init=init, target=i, vars=["global", "shift"], params=params,n_chain=n_chain, verbose=0))
    models_local .append(FlexibleFitting(init=init, target=i, vars=["local", "shift"], params=params,n_chain=n_chain, verbose=0))
    models_glocal.append(FlexibleFitting(init=init, target=i, vars=["local", "global", "shift"], params=params,n_chain=n_chain, verbose=0))

# n_test = 40
# models[n_test].HMC_chain()
# models[n_test].show()
# models[n_test].show_3D()
fits_global = multiple_fitting(models_global, n_chain=n_chain, n_proc=n_proc)
fits_local = multiple_fitting(models_local, n_chain=n_chain, n_proc=n_proc)
fits_glocal = multiple_fitting(models_glocal, n_chain=n_chain, n_proc=n_proc)

n=39
chimera_fit_viewer(fits_glocal[n].res["mol"], targets[n])
# chimera_molecule_viewer([mols[0]]), mols[10], mols[20], mols[30], mols[40]])

pca_data = [i.coords.flatten() for i in mols] + [i.res["mol"].coords.flatten() for i in fits_global]+ \
                                                [i.res["mol"].coords.flatten() for i in fits_local]+ \
                                                [i.res["mol"].coords.flatten() for i in fits_glocal]
length= [N,N,N,N]
labels=["Ground Truth", "Global", "Local", "Global + Local"]
compute_pca(data=pca_data, length=length, labels= labels, n_components=2)



############################################################
###  TEMP, kINETIC ETC
############################################################
import matplotlib.pyplot as plt
import numpy as np

from src.molecule import Molecule
from src.density import Volume
from src.constants import *
import src.forcefield

init = Molecule.from_file("data/AK/AK_prody/output.pdb")
init.center_structure()
fnModes = np.array(["data/AK/AK_prody/modes_psf/vec."+str(i+7) for i in range(6)])
init.add_modes(fnModes)
# init.select_atoms(pattern="CA")
init.set_forcefield("data/AK/AK_prody/output.psf")
voxel_size=2.0
init_density = Volume.from_coords(coord=init.coords, voxel_size=voxel_size, size=64, threshold=5, sigma=2)


T=300
m = (init.prm.mass*ATOMIC_MASS_UNIT)   # kg
sigma = (np.ones((3,init.n_atoms)) * np.sqrt((K_BOLTZMANN *T)/ (init.prm.mass*ATOMIC_MASS_UNIT))*1e10).T
v = np.random.normal(0,sigma, init.coords.shape)          # A * s-1
K = src.forcefield.get_kinetic(v, init.prm.mass)   # kg * A2 * s-2
K_J = K*1e-20
K_kcalmol =AVOGADRO_CONST* K_J /KCAL_TO_JOULE

T_hat =  src.forcefield.get_instant_temp(K, init.n_atoms)   # K
print(T_hat)

U_kcalmol= init.get_energy() # kcal * mol-1
U_jmol = (KCAL_TO_JOULE * U_kcalmol) # J * mol-1
U_j = U_jmol/AVOGADRO_CONST # J

F_kcalmolA = src.forcefield.get_autograd(params={"local":np.zeros(init.coords.shape)}, mol=init)["local"] # kcal * mol-1 * A-1
F_JA = (KCAL_TO_JOULE * F_kcalmolA) / AVOGADRO_CONST                                             # kg * m2 * s-2 - A-1
F = F_JA * (1e10)**2  #kg * A * s-2
a = (F.T*(1/m)).T      #A * s-2

H=U+K


f1 = K_J/U_j
f2 = K_kcalmol/U_kcalmol




import os
import matplotlib.pyplot as plt
import numpy as np

from src.molecule import Molecule
from src.density import Volume
from src.constants import *
from src.simulation import nma_deform
import src.forcefield
from src.viewers import chimera_molecule_viewer

pdb = "data/AK/AK_prody/output.pdb"
modes = "/home/guest/ScipionUserData/projects/nma_calculator/Runs/000839_FlexProtNMA/modes2.xmd"
new_pdb = "mol.pdb"
q= np.zeros(14)
q[0] = 300
q[2] = -100


mol1 = Molecule.from_file(pdb)
fnModes = np.array(["/home/guest/ScipionUserData/projects/nma_calculator/Runs/000839_FlexProtNMA/modes/vec."+str(i+7) for i in range(len(q))])
mol1.add_modes(fnModes)
mol1 = nma_deform(mol1, q)
mol1.show()
mol1.save_pdb(file= "data/AK/AK_prody/AK1.pdb")

cmd = "/home/guest/xmipp-bundle/xmipp/build/bin/xmipp_pdb_nma_deform"
cmd +=" --pdb "+ pdb
cmd +=" --nma "+ modes
cmd +=" -o "+ new_pdb
cmd +=" --deformations "+ ' '.join(str(i) for i in q)
print(cmd)

mol2 = Molecule.from_file(new_pdb)
chimera_molecule_viewer([mol1, mol2])




from src.molecule import Molecule
from src.simulation import nma_deform
from src.flexible_fitting import *
from src.viewers import molecule_viewer, chimera_molecule_viewer, chimera_fit_viewer
from src.density import Volume
from src.constants import *

########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
init = Molecule.from_file("data/1AKE/1ake_chainA_psf.pdb")
# init.set_forcefield(psf_file="data/1AKE/1ake_chainA.psf")
init.select_atoms()
init.set_forcefield()
init.center_structure()

target =  Molecule.from_file("data/4AKE/4ake_fitted.pdb")
target.center_structure()
target.select_atoms()

size=64
sampling_rate=2.2
threshold= 4
gaussian_sigma=2
target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold)
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold)
target_density.show()

# chimera_molecule_viewer([target, init])
# chimera_fit_viewer(init, target_density)

########################################################################################################
#               HMC
########################################################################################################
params ={
    "biasing_factor" : 100,
    "n_step": 20,

    "local_dt" : 4*1e-15,
    "temperature" : 1000,

    "global_dt" : 0.1,
    "rotation_dt" : 0.00001,
    "n_iter":30,
    "n_warmup":15,
}


fit  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params, n_chain=4, verbose=2)

fit.HMC()
fit.show()
fit.show_3D()
chimera_molecule_viewer([fit.res["mol"], target])

data= []
for j in [[i.flatten() for i in n["coord"]] for n in fit.fit]:
    data += j
src.functions.compute_pca(data=data, length=[len(data)], n_components=2)