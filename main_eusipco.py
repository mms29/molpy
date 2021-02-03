import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting
from src.viewers import structures_viewer, chimera_structure_viewer
from src.flexible_fitting import *
import src.molecule
import src.io
import pickle
import argparse

cc1 =[]
cc2 =[]
cc3 =[]


for i in range(20):
    with open("results/EUSIPCO/HMCNMA"+str(i+1)+".pkl", "rb") as f:
        dic = pickle.load(file=f)

        l1 = np.cumsum(dic["NMA"]["l"])
        l2 = np.cumsum(dic["NMAHMC"]["l"])
        l3 = np.cumsum(dic["HMC"]["l"])

        cc1.append(np.array(dic["NMA"]["cc"])[l1-1])
        cc2.append(np.array(dic["NMAHMC"]["cc"])[l2-1])
        cc3.append(np.array(dic["HMC"]["cc"])[l3-1])

fig, ax = plt.subplots(1,1)
ax.plot(np.mean(np.array(cc1), axis=0))
ax.plot(np.mean(np.array(cc2), axis=0))
ax.plot(np.mean(np.array(cc3), axis=0))