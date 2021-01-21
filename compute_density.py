import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting
import numpy as np
import time


# import PDB
atoms, ca = src.functions.read_pdb("data/AK/AK.pdb")
modes = src.functions.read_modes("data/AK/modes/vec.", n_modes=5)[ca][:]
atoms= src.functions.center_pdb(atoms)[ca][:]
n_atoms=atoms.shape[0]

size=32
gaussian_sigma = 2
sampling_rate = 3

# t_slow= time.time()
# density_slow = volume_from_pdb_slow(atoms, size, gaussian_sigma, sampling_rate)
# dt_slow = time.time() - t_slow
# print("slow="+str(dt_slow))

# t_vec = time.time()
# density_vec = volume_from_pdb(atoms, size, gaussian_sigma, sampling_rate)
# dt_vec = time.time() - t_vec
# print("vec="+str(dt_vec))
#
# t_fast = time.time()
# density_fast = volume_from_pdb_fast(atoms, size, gaussian_sigma, sampling_rate)
# dt_fast = time.time() - t_fast
# print("fast="+str(dt_fast))

t_fast2 = time.time()
density_fast2 = volume_from_pdb_fast2(atoms, size, gaussian_sigma, sampling_rate)
dt_fast2 = time.time() - t_fast2
print("fast2="+str(dt_fast2))
plt.figure()
plt.imshow(density_fast2[int(size/2)])

t_fast3 = time.time()
density_fast3 = volume_from_pdb_fast3(atoms, size, gaussian_sigma, sampling_rate, threshold=3)
dt_fast3 = time.time() - t_fast3
print("fast3="+str(dt_fast3))
plt.figure()
plt.imshow(density_fast3[int(size/2)])

print("err="+str(np.sum(np.square(density_fast2-density_fast3))))


