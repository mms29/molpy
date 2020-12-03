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

N=16
gaussian_sigma = 2
sampling_rate = 6

t_slow= time.time()
density_slow = volume_from_pdb_slow(atoms, N, gaussian_sigma, sampling_rate)
dt_slow = time.time() - t_slow
print(dt_slow)

t_vec = time.time()
density_vec = volume_from_pdb(atoms, N, gaussian_sigma, sampling_rate)
dt_vec = time.time() - t_vec
print(dt_vec)

t_fast = time.time()
density_fast = volume_from_pdb_fast(atoms, N, gaussian_sigma, sampling_rate)
dt_fast = time.time() - t_fast
print(dt_fast)

np.sum(density_fast-density_vec)