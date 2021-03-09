import numpy as np
import matplotlib.pyplot as plt
from src.functions import *
from src.io import read_pdb
import src.constants
import src
import src.forcefield
import src.molecule
import src.flexible_fitting


ak= src.io.read_pdb("data/AK/AK.pdb")
ak = ak.select_atoms()

U = ak.get_energy()
F = ak.get_gradient()

dt= 0.0001
molstep, fstep = src.flexible_fitting.run_md(mol=ak, dt=dt, total_time=dt * 1000, temperature=1000)

plot_structure([molstep[0].coords, molstep[-1].coords])

#
# N=200
# first  =np.array([[ 2.18058707, -2.99928724,  7.95602422],
#        [ 0.24942126, -6.23103142,  8.44624217],
#        [ 2.54902553, -8.83097488,  6.90806155]])
#
# bonds = np.ones(N-3) * src.constants.R0_BONDS            + np.random.normal(0,1e-2,N-3)
# angles= np.ones(N-3) * src.constants.THETA0_ANGLES       + np.random.normal(0,1e-2,N-3)
# torsions= np.random.randint(0,2,N-3)*-np.pi + np.pi/4    + np.random.normal(0,1e-10,N-3)
#
# # bonds[10] = 5
# # bonds[100] = 3
# # angles[10] = 1.8
# # angles[100] = 0.5
# # angles[101] = 1.1
# # torsions[25] = -2
# # torsions[50] = -2.2
# # torsions[100] = -2.5
# # torsions[125] = -3
# test = internal_to_cartesian(np.array([bonds, angles, torsions]).T, first)
# mol_test = src.molecule.Molecule.from_coords(test)
# # mol_test.show()
#
# U = mol_test.get_energy()
# G = mol_test.get_gradient()
#
# G_bonds = np.linalg.norm(src.force_field.get_bonds_gradient(mol_test), axis=1)
# plt.plot(G_bonds*0.001)
# plt.plot(np.arange(N-3)+3 -0.5, np.square(bonds - src.constants.R0_BONDS))
#
# G_angles = np.linalg.norm(src.force_field.get_angles_gradient(mol_test), axis=1)
# plt.plot(G_angles*0.01)
# plt.plot(np.arange(N-3)+3 -1, np.square(angles - src.constants.THETA0_ANGLES))
#
# G_torsions = np.linalg.norm(src.force_field.get_torsions_gradient(mol_test), axis=1)
# plt.plot(G_torsions*0.1)
# plt.plot(np.arange(N-3)+3 -1, 1+np.cos(src.constants.N_TORSIONS*torsions-src.constants.DELTA_TORSIONS))
#
# plt.hist(ak.torsions , 100)
# x=np.arange(-np.pi,np.pi,0.01)
# c = src.constants.K_TORSIONS*(1+ np.cos(src.constants.N_TORSIONS*x -src.constants.DELTA_TORSIONS))
# plt.plot(x,c)
# plt.axvline(np.pi)
# plt.axvline((np.pi)/4)
# plt.axvline(-np.pi)
# plt.axvline(-np.pi*3/4)
#
# plt.hist(ak.angles , 100)
# x=np.arange(-0,np.pi,0.01)
# c = src.constants.K_ANGLES * np.square(x - src.constants.THETA0_ANGLES)
# plt.plot(x,c)
#
# plt.hist(ak.bonds , 100)
# x=np.arange(-0,10,0.01)
# c = src.constants.K_BONDS * np.square(x - src.constants.R0_BONDS)
# plt.plot(x,c)