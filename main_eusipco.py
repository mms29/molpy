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

l1 =[]
l2 =[]
l3 =[]


for i in range(20):
    with open("results/EUSIPCO2/HMCNMA"+str(i+1)+".pkl", "rb") as f:
        dic = pickle.load(file=f)

        l1.append(np.cumsum([1]+dic["NMA"]["l"])-1)
        l2.append(np.cumsum([1]+dic["NMAHMC"]["l"])-1)
        l3.append(np.cumsum([1]+dic["HMC"]["l"])-1)

        cc1.append(np.array([0.76]+dic["NMA"]["cc"])[l1[-1]])
        cc2.append(np.array([0.76]+dic["NMAHMC"]["cc"])[l2[-1]])
        cc3.append(np.array([0.76]+dic["HMC"]["cc"])[l3[-1]])

cc1 = np.array(cc1)
cc2 = np.array(cc2)
cc3 = np.array(cc3)
l1 = np.array(l1)
l2 = np.array(l2)
l3 = np.array(l3)

fig, ax = plt.subplots(1,1, figsize=(6,3))
ax.errorbar(np.mean(l2,axis=0), np.mean(cc2, axis=0), fmt='--o', color='tab:blue',  capsize=5,capthick=1,label="NMA/HMC")#, yerr=np.sqrt(np.var(cc2, axis=0)*20),xerr=np.sqrt(np.var(l2, axis=0)))
ax.errorbar(np.mean(l3,axis=0), np.mean(cc3, axis=0), fmt='--o', color='tab:green', capsize=5,capthick=1,label="HMC"    )#, yerr=np.sqrt(np.var(cc3, axis=0)*20),xerr=np.sqrt(np.var(l3, axis=0)))
ax.set_ylabel("Cross Correlation")
ax.set_xlabel("Integrator Steps")
ax.legend()
fig.tight_layout()
# fig.savefig('results/EUSIPCO/HMCNMA_0.4.png', format='png', dpi=1000)

