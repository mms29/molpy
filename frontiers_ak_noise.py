
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

# protocol_list =[
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/000957_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001176_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001387_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001709_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001928_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/002450_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/002679_FlexProtGenesisFit/",
#
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001031_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001250_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001461_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/001783_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/002002_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/002524_FlexProtGenesisFit/",
# "/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/002753_FlexProtGenesisFit/",
# ]
# labels=["5A","7A","9A","11A","13A", "15A", "17A",
#         "5A","7A","9A","11A","13A", "15A", "17A"]
#
# colors = list(pl.cm.Reds(np.linspace(0.5,1,len(protocol_list)//2))) + list(pl.cm.Blues(np.linspace(0.5,1,len(protocol_list)//2)))
#
#
# fig, ax = plt.subplots(1, 2, figsize=(10,3))
# for i in range(len(protocol_list)):
#     cc = []
#     rmsd = []
#     molp=[]
#     cc = np.load(protocol_list[i]+ "/extra/run_r1_cc.npy")
#     rmsd = np.load(protocol_list[i] +"/extra/run_r1_rmsd.npy")
#     t = np.arange(len(cc)) * 100 * 0.002
#     print( "lent" + str(len(t)))
#     print( "lenrmsd" + str(len(rmsd)))
#
#     ax[0].plot(t, cc, label=labels[i], color=colors[i])
#     ax[1].plot(t, rmsd, label=labels[i], color=colors[i])
#
# ax[0].set_xlabel("Simulation Time (ps)")
# ax[0].set_ylabel("CC")
# ax[0].set_title("Correlation Coefficient")
# ax[1].set_xlabel("Simulation Time (ps)")
# ax[1].set_ylabel("RMSD (A)")
# ax[1].set_title("Root Mean Square Deviation")
# fig.tight_layout()
# handles, labels = ax[0].get_legend_handles_labels()
# fig.legend(handles = handles, loc='lower right')



protocol_list =[
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/003358_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/003717_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004014_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004311_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004564_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004941_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/005198_FlexProtGenesisFit/",

"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/003526_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/003791_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004088_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004385_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/004638_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/005015_FlexProtGenesisFit/",
"/home/guest/ScipionUserData/projects/PaperFrontiers_AK_noise/Runs/005272_FlexProtGenesisFit/",
]
labels=["1", "0.8", "0.6", "0.4", "0.2", "0.1", "0.8",
        "1", "0.8", "0.6", "0.4", "0.2", "0.1", "0.8"]

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
