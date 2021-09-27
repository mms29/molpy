
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
import os.path

protocol_list =[
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/000762_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/000957_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001176_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001387_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001709_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001928_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/002450_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/002679_FlexProtGenesisFit/",
#
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/000838_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001031_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001250_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001461_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001783_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/002002_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/002524_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/002753_FlexProtGenesisFit/",
]
labels=["2A","4A","6A","8A","10A",
        "2A","4A","6A","8A","10A"]

snr = [2,4,6,8,10]

colors = list(pl.cm.Reds(np.linspace(0.5,1,len(protocol_list)//2))) + list(pl.cm.Blues(np.linspace(0.5,1,len(protocol_list)//2)))

fig, ax = plt.subplots(1, 2, figsize=(10,3))
step=10
rmsd_all=[]
for i in range(len(protocol_list)):
    cc = []
    rmsd = []
    molp=[]
    for j in range(10):
        print(protocol_list[i]+ "/extra/run%i_cc.npy"%(j+1))
        cc.append(np.load(protocol_list[i]+ "/extra/run%i_cc.npy"%(j+1)))
        rmsd.append(np.load(protocol_list[i] +"/extra/run%i_rmsd.npy"%(j+1)))
    cc= np.array(cc)
    rmsd= np.array(rmsd)
    rmsd_all.append(rmsd)
    t = np.arange(cc.shape[1]) * 100 * 0.002

    ax[0].plot(t, cc.mean(axis=0), color=colors[i])
    ax[0].errorbar(x=t[::step], y=cc.mean(axis=0)[::step], yerr=cc.std(axis=0)[::step],
                      label=labels[i], color=colors[i], fmt="o",
                      capthick=1.7, capsize=5,elinewidth=1.7)
    ax[1].plot(t, rmsd.mean(axis=0), color=colors[i])
    ax[1].errorbar(x=t[::step], y=rmsd.mean(axis=0)[::step], yerr=rmsd.std(axis=0)[::step],
                      label=labels[i], color=colors[i], fmt="d",
                      capthick=1.7, capsize=5,elinewidth=1.7)

ax[0].set_xlabel("Simulation Time (ps)")
ax[0].set_ylabel("CC")
ax[0].set_title("Correlation Coefficient")
ax[1].set_xlabel("Simulation Time (ps)")
ax[1].set_ylabel("RMSD (A)")
ax[1].set_title("Root Mean Square Deviation")
fig.tight_layout()
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles = handles, loc='lower right')

rmsd_all_mmin_mean = np.mean(np.min(rmsd_all, axis=(2)), axis=1)
rmsd_all_mmin_std = np.std(np.min(rmsd_all, axis=(2)), axis=1)


rmsd_all = np.array(rmsd_all)
rmsd_all.shape
period=100
clock_time=540
#RMSD Aver
rmsd_min= np.min(rmsd_all,axis=2)
rmsdtime= []
for p in range(len(protocol_list)):
    rmsdclocktimeMean = []
    for i in range(step):
        rmsd5pMean = rmsd_all[p,i].min() + 0.01 * (rmsd_all[p,i].max() - rmsd_all[p,i].min())
        rmsdtimeMean = np.min(np.where(rmsd_all[p,i] <= rmsd5pMean)[0]) * period * 0.002
        rmsdclocktimeMean.append((rmsdtimeMean / ((len(rmsd_all[p,i]) - 1) * period * 0.002)) * clock_time)
    rmsdtime.append(rmsdclocktimeMean)

rmsdtimemean = np.mean(rmsdtime, axis=1)
rmsdtimestd = np.std(rmsdtime, axis=1)

fig, ax = plt.subplots(1,2,figsize=(10,3))
ax[0].errorbar(x=snr,y=rmsdtimemean[:len(snr)],yerr=rmsdtimestd[:len(snr)],fmt="o-",
                       capthick=1.7, capsize=5,elinewidth=1.7, color="tab:red", label="MD")
ax[0].errorbar(x=snr,y=rmsdtimemean[len(snr):],yerr=rmsdtimestd[len(snr):],fmt="o-",
                       capthick=1.7, capsize=5,elinewidth=1.7, color="tab:blue", label="NMMD")
ax[1].errorbar(x=snr,y=rmsd_all_mmin_mean[:len(snr)],yerr=rmsd_all_mmin_std[:len(snr)],fmt="o-",
                       capthick=1.7, capsize=5,elinewidth=1.7, color="tab:red", label="MD")
ax[1].errorbar(x=snr,y=rmsd_all_mmin_mean[len(snr):],yerr=rmsd_all_mmin_std[len(snr):],fmt="d-",
                       capthick=1.7, capsize=5,elinewidth=1.7, color="tab:blue", label="NMMD")
ax[0].set_xlabel(r"Resolution ($\AA$)")
ax[1].set_xlabel(r"Resolution ($\AA$)")
ax[0].set_ylabel("Convergence Time (s)")
ax[1].set_ylabel(r"RMSD ($\AA$)")
# ax[0].set_xscale("log")
# ax[1].set_xscale("log")
ax[1].legend()
fig.tight_layout()
fig.savefig("/home/guest/Pictures/resolution.png", dpi=1000)

protocol_list =[
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004014_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004311_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004564_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004941_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/005198_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/005451_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/005740_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/006715_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/007008_FlexProtGenesisFit/",

# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004088_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004385_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004638_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/005015_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/005272_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/005525_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/005814_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/006787_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/007080_FlexProtGenesisFit/",
]
labels=["0.05", "0.03", "0.01", "0.007", "0.003","0.001","0.00075","0.0005",
        "0.05", "0.03", "0.01", "0.007", "0.003","0.001","0.00075","0.0005",]
snr = [0.05, 0.03, 0.01, 0.008, 0.006,0.004,0.002,0.001]

colors = list(pl.cm.Reds(np.linspace(0.5,1,len(protocol_list)//2))) + list(pl.cm.Blues(np.linspace(0.5,1,len(protocol_list)//2)))

fig, ax = plt.subplots(1, 2, figsize=(10,3))
for i in range(len(protocol_list)):
    cc = []
    rmsd = []
    molp=[]
    for j in range(5):
        cc.append(np.load(protocol_list[i]+ "/extra/run_r%i_cc.npy"%(j+1)))
        rmsd.append(np.load(protocol_list[i] +"/extra/run_r%i_rmsd.npy"%(j+1)))
    cc = np.mean(cc, axis=0)
    rmsd = np.mean(rmsd, axis=0)
    t = np.arange(len(cc)) * 100 * 0.002

    ax[0].plot(t, cc, label=labels[i], color=colors[i])
    ax[1].plot(t, rmsd, label=labels[i], color=colors[i])

ax[0].set_xlabel("Simulation Time (ps)")
ax[0].set_ylabel("CC")
ax[0].set_title("Correlation Coefficient")
ax[1].set_xlabel("Simulation Time (ps)")
ax[1].set_ylabel("RMSD (A)")
ax[1].set_title("Root Mean Square Deviation")
fig.tight_layout()
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles = handles, loc='lower right')
