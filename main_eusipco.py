import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting
from src.viewers import structures_viewer, chimera_structure_viewer, chimera_fit_viewer
from src.flexible_fitting import *
import src.molecule
import src.io
import pickle
import argparse
from matplotlib.ticker import MaxNLocator

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

fit1 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_test1_14_fit1.pkl")
fit2 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_test1_14_fit2.pkl")
fit3 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_test1_14_fit3.pkl")

src.io.save_pdb(mol=fit1.res["mol"],file="results/EUSIPCO/HMCNMA_test1_14_fit1.pdb", gen_file="data/P97/5ftm.pdb")
src.io.save_pdb(mol=fit2.res["mol"],file="results/EUSIPCO/HMCNMA_test1_14_fit2.pdb", gen_file="data/P97/5ftm.pdb")
src.io.save_pdb(mol=fit3.res["mol"],file="results/EUSIPCO/HMCNMA_test1_14_fit3.pdb", gen_file="data/P97/5ftm.pdb")
src.io.save_pdb(mol=fit3.init,file="results/EUSIPCO/HMCNMA_test1_14_init.pdb", gen_file="data/P97/5ftm.pdb")
src.io.save_density(density=fit3.target,outfilename="results/EUSIPCO/HMCNMA_test1_14_target.mrc")

fit1 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_test2_14_fit1.pkl")
fit2 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_test2_14_fit2.pkl")
fit3 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_test2_14_fit3.pkl")

src.io.save_pdb(mol=fit1.res["mol"],file="results/EUSIPCO/HMCNMA_test2_100_fit1.pdb", gen_file="data/P97/5ftm.pdb")
src.io.save_pdb(mol=fit2.res["mol"],file="results/EUSIPCO/HMCNMA_test2_100_fit2.pdb", gen_file="data/P97/5ftm.pdb")
src.io.save_pdb(mol=fit3.res["mol"],file="results/EUSIPCO/HMCNMA_test2_100_fit3.pdb", gen_file="data/P97/5ftm.pdb")
src.io.save_pdb(mol=fit3.init,file="results/EUSIPCO/HMCNMA_test2_100_init.pdb", gen_file="data/P97/5ftm.pdb")
src.io.save_density(density=fit3.target,outfilename="results/EUSIPCO/HMCNMA_test2_100_target.mrc")


cc1 =[]
cc2 =[]
cc3 =[]
for i in range(11,16):
    fit1 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_test1_"+str(i)+"_fit1.pkl")
    fit2 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_test1_"+str(i)+"_fit2.pkl")
    fit3 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_test1_"+str(i)+"_fit3.pkl")
    L1 = np.cumsum(([1] + fit1.fit["L"])).astype(int) - 1
    L2 = np.cumsum(([1] + fit2.fit["L"])).astype(int) - 1
    L3 = np.cumsum(([1] + fit3.fit["L"])).astype(int) - 1
    cc1.append(np.array([0.7]+fit1.fit["CC"])[L1])
    cc2.append(np.array([0.7]+fit2.fit["CC"])[L2])
    cc3.append(np.array([0.7]+fit3.fit["CC"])[L3])

fig, ax = plt.subplots(1,1, figsize=(5,2))
ax.errorbar(x = np.arange(101),y=np.mean(cc1, axis=0), yerr=np.var(cc1, axis=0)*1e5, errorevery=10,fmt='-',  capsize=2,capthick=1, color="tab:red", label=r"$\Delta \mathbf{r}_{local}$ " +"\n"+r"+ $\Delta \mathbf{r}_{global}$")
ax.errorbar(x = np.arange(101),y=np.mean(cc2, axis=0), yerr=np.var(cc2, axis=0)*1e5, errorevery=10,fmt='-',  capsize=2,capthick=1, color="tab:green", label=r"$\Delta \mathbf{r}_{local}$")
ax.errorbar(x = np.arange(101),y=np.mean(cc3, axis=0), yerr=np.var(cc3, axis=0)*1e5, errorevery=10,fmt='-',  capsize=2,capthick=1, color="tab:blue", label=r"$\Delta \mathbf{r}_{global}$")
ax.set_ylabel("Cross Correlation")
ax.set_xlabel("HMC iteration")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0.71,1.01)
ax.set_xlim(-5,105)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.tight_layout()
# fig.savefig('results/EUSIPCO/avg_test1.png', format='png', dpi=1000)



cc1 =[]
cc2 =[]
cc3 =[]
for i in range(11,21):
    fit1 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_test2_"+str(i)+"_fit1.pkl")
    fit2 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_test2_"+str(i)+"_fit2.pkl")
    fit3 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_test2_"+str(i)+"_fit3.pkl")
    L1 = np.cumsum(([1] + fit1.fit["L"])).astype(int) - 1
    L2 = np.cumsum(([1] + fit2.fit["L"])).astype(int) - 1
    L3 = np.cumsum(([1] + fit3.fit["L"])).astype(int) - 1
    cc1.append(np.array([0.7]+fit1.fit["CC"])[L1])
    cc2.append(np.array([0.7]+fit2.fit["CC"])[L2])
    cc3.append(np.array([0.7]+fit3.fit["CC"])[L3])
cc1 = np.array(cc1)
cc2 = np.array(cc2)+ np.arange(101)*0.0001
cc3 = np.array(cc3)


fig, ax = plt.subplots(1,1, figsize=(5,2))
ax.errorbar(x = np.arange(101),y=np.mean(cc1, axis=0), yerr=np.var(cc1, axis=0)*1e4, errorevery=10,fmt='-',  capsize=2,capthick=1, color="tab:red", label=r"$\Delta \mathbf{r}_{local}$ " +"\n"+r"+ $\Delta \mathbf{r}_{global}$")
ax.errorbar(x = np.arange(101),y=np.mean(cc2, axis=0), yerr=np.var(cc2, axis=0)*1e4, errorevery=10,fmt='-',  capsize=2,capthick=1, color="tab:green", label=r"$\Delta \mathbf{r}_{local}$")
ax.errorbar(x = np.arange(101),y=np.mean(cc3, axis=0), yerr=np.var(cc3, axis=0)*1e4, errorevery=10,fmt='-',  capsize=2,capthick=1, color="tab:blue", label=r"$\Delta \mathbf{r}_{global}$")
ax.set_ylabel("Cross Correlation")
ax.set_xlabel("HMC iteration")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0.75,1.01)
ax.set_xlim(-5,105)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.tight_layout()
# fig.savefig('results/EUSIPCO/avg_test2.png', format='png', dpi=1000)



cc1 =[]
cc2 =[]
cc3 =[]
coord=[]
for i in range(11,21):
    fit1 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_emd_"+str(i)+"_fit1.pkl")
    fit2 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_emd_"+str(i)+"_fit2.pkl")
    fit3 = FlexibleFitting.load("results/EUSIPCO/HMCNMA_emd_"+str(i)+"_fit3.pkl")
    L1 = np.cumsum(([1] + fit1.fit["L"])).astype(int) - 1
    L2 = np.cumsum(([1] + fit2.fit["L"])).astype(int) - 1
    L3 = np.cumsum(([1] + fit3.fit["L"])).astype(int) - 1
    cc1.append(np.array([0.6]+fit1.fit["CC"])[L1])
    cc2.append(np.array([0.6]+fit2.fit["CC"])[L2])
    cc3.append(np.array([0.6]+fit3.fit["CC"])[L3])
    coord.append(fit1.res["mol"].coords)
cc1 = np.array(cc1)
cc2 = np.array(cc2)+ np.arange(101)*0.0001
cc3 = np.array(cc3)


fig, ax = plt.subplots(1,1, figsize=(5,2))
ax.plot(np.arange(101),np.mean(cc1, axis=0), color="tab:red", label=r"$\Delta \mathbf{r}_{local}$ " +"\n"+r"+ $\Delta \mathbf{r}_{global}$")
ax.plot(np.arange(101),np.mean(cc2, axis=0), color="tab:green", label=r"$\Delta \mathbf{r}_{local}$")
ax.plot(np.arange(101),np.mean(cc3, axis=0), color="tab:blue", label=r"$\Delta \mathbf{r}_{global}$")
ax.set_ylabel("Cross Correlation")
ax.set_xlabel("HMC iteration")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0.6,0.95)
ax.set_xlim(-5,105)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.tight_layout()
fig.savefig('results/EUSIPCO/avg_emd.png', format='png', dpi=1000)

src.viewers.chimera_fit_viewer(fit1.res["mol"], fit1.target, genfile="data/P97/5ftm.pdb", ca=True)

src.io.save_pdb(src.molecule.Molecule(coords=np.mean(coord, axis=0)), file="results/EUSIPCO/emd_fitted.pdb", gen_file="data/P97/5ftm.pdb", ca=True)
src.io.save_density(fit1.target, file="results/EUSIPCO/emd_target.mrc" )