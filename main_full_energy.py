from src.io import *
from src.constants import *
from itertools import permutations
from src.molecule import MoleculeForcefieldPrm
from src.forcefield import *
import matplotlib.pyplot as plt
import time

init = read_pdb("data/P97/5ftm_PSF.pdb")
init.set_forcefield(psf_file="data/P97/5ftm.psf", prm_file=PARAMETER_FILE)

################################################################
#       BONDS
################################################################

U_bonds = get_U_bonds(init.coords,init.bonds, init.prm)

t = time.time()
F_bonds = get_F_bonds_auto(init.coords, init.bonds, init.prm)
print(time.time() - t)

t = time.time()
# F_bonds2 = get_F_bonds(init.coords, psf["bonds"], prm)
print(time.time() - t)

# print(np.max(F_bonds2- F_bonds))

r = np.linalg.norm(init.coords[init.bonds[:, 0]] - init.coords[init.bonds[:, 1]], axis=1)

plt.figure()
plt.hist(r,100)
plt.hist(init.prm.b0,100)
# plt.xlim(0,180)


################################################################
#       ANGLES
################################################################

U_angles = get_U_angles(init.coords, init.angles, init.prm)

t = time.time()
F_angles = get_F_angles_auto(init.coords,init.angles, init.prm)
print(time.time() - t)

a1 = init.coords[init.angles[:, 2]]
a2 = init.coords[init.angles[:, 1]]
a3 = init.coords[init.angles[:, 0]]
theta = -np.arccos(np.sum((a1 - a2) * (a2 - a3), axis=1)
                   / (np.linalg.norm(a1 - a2, axis=1) * np.linalg.norm(a2 - a3, axis=1))) + np.pi
plt.figure()
plt.hist(theta*180/np.pi,100)
plt.hist(init.prm.Theta0,100)
plt.xlim(0,180)

################################################################
#       DIHEDRALS
################################################################

U_dihedrals = get_U_dihedrals(init.coords, init.dihedrals, init.prm)
F_dihedrals = get_F_dihedrals_auto(init.coords, init.dihedrals, init.prm)

coord= init.coords
u1 = coord[init.dihedrals[:, 1]] - coord[init.dihedrals[:, 0]]
u2 = coord[init.dihedrals[:, 2]] - coord[init.dihedrals[:, 1]]
u3 = coord[init.dihedrals[:, 3]] - coord[init.dihedrals[:, 2]]
torsions = npg.arctan2(npg.linalg.norm(u2, axis=1) * npg.sum(u1 * npg.cross(u2, u3), axis=1),
                       npg.sum(npg.cross(u1, u2) * npg.cross(u2, u3), axis=1))

plt.figure()
plt.hist(torsions,100)
plt.hist(init.prm.delta*np.pi/180,100)
x = np.arange(-3.14,3.14,0.1)
plt.plot(x, np.cos(x*3)*1000)
plt.plot(x, np.cos(x*2)*1000)
plt.plot(x, np.cos(x*6)*1000)
plt.plot(x, np.cos(x)*1000)



################################################################
#       TOTAL
################################################################

get_U(init.coords, psf, prm)

t = time.time()
f= get_F_auto(init, init.coords)
print(time.time() - t)


t = time.time()
f = get_F_auto(init, init.coords)
print(time.time() - t)




