import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting
import numpy as np
import time


init =src.io.read_pdb("data/P97/5ftm.pdb")
init.center_structure()
# init = init.select_atoms(pattern='CA')

size=128
sampling_rate=2
threshold=4
gaussian_sigma=2
coord= init.coords

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

# t_fast2 = time.time()
# density_fast2 = volume_from_pdb_fast2(coord, size, gaussian_sigma, sampling_rate)
# dt_fast2 = time.time() - t_fast2
# print("fast2="+str(dt_fast2))
# plt.figure()
# plt.imshow(density_fast2[int(size/2)])

t_fast3 = time.time()
density_fast3 = volume_from_pdb_fast3(coord, size, gaussian_sigma, sampling_rate, threshold=3)
dt_fast3 = time.time() - t_fast3
print("fast3="+str(dt_fast3))
plt.figure()
plt.imshow(density_fast3[int(size/2)])

t_fast4 = time.time()
density_fast4 = volume_from_pdb_fast4(coord, size, gaussian_sigma, sampling_rate, threshold=3)
dt_fast4 = time.time() - t_fast4
print("fast4="+str(dt_fast4))

print("err="+str(np.sum(np.square(density_fast4-density_fast3))))

t = time.time()
get_grad_RMSD3(coord,density_fast4,density_fast4, size, gaussian_sigma, sampling_rate, threshold=3)
print(time.time()-t)

# IMAGES

t = time.time()
img0 = image_from_pdb(coord=init.coords, size=size, sampling_rate=sampling_rate, sigma=gaussian_sigma)
t0 = time.time()- t
print("t0="+str(t0))

t = time.time()
img1 = image_from_pdb_fast(coord=init.coords, size=size, sampling_rate=sampling_rate, sigma=gaussian_sigma)
t1 = time.time()- t
print("t1="+str(t1))


t = time.time()
img2 = image_from_pdb_fast2(coord=init.coords, size=size, sampling_rate=sampling_rate, sigma=gaussian_sigma)
t2 = time.time()- t
print("t2="+str(t2))


t = time.time()
img3 = image_from_pdb_fast3(coord=init.coords, size=size, sampling_rate=sampling_rate, sigma=gaussian_sigma, threshold=10)
t3 = time.time()- t
print("t3="+str(t3))

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img3)