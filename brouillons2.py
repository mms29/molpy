
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

def show_cc_rmsd(protocol_list, length, labels=None, period=100, step=10,
                 fvar=10.0, capthick=10.0, capsize=10.0, elinewidth=1.0, figsize=(10,5), dt=0.002,
                 colors=["tab:blue", "tab:red", "tab:green", "tab:orange",
                         "tab:brown", "tab:olive", "tab:pink", "tab:green", "tab:cyan"],
                 fmts = ["o", "d", "v", "^", "p", "*", "s","x"], end=-1, start=0):
    if labels is None:
        labels = ["#"+str(i) for i in range(len(protocol_list))]
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    for i in range(len(protocol_list)):
        cc = []
        rmsd = []
        molp=[]
        for j in range(start, length[i]):
            cc.append(np.load(protocol_list[i]+ "/extra/run_r"+str(j+1)+"_cc.npy")[:end])
            rmsd.append(np.load(protocol_list[i] +"/extra/run_r"+str(j+1)+"_rmsd.npy")[:end])
            # molp.append(np.load(protocol_list[i] +"/extra/run_r"+str(j+1)+"_molprobity.npy"))
            t = np.arange(len(cc[-1])) * period * dt
            print( "lent" + str(len(t)))
            print( "lenrmsd" + str(len(rmsd[-1])))
            ax[0].plot(t, cc[-1], "-", color=colors[i], alpha=0.5)
            ax[1].plot(t, rmsd[-1], "-", color=colors[i], alpha=0.5)
            # ax[1,0].plot(i, molp[-1][0], "o", color=colors[i])
            # ax[1,1].plot(i, molp[-1][1], "o", color=colors[i])
        ax[0].errorbar(x=t[::step], y=np.mean(cc, axis=0)[::step],  yerr=np.var(cc, axis=0)[::step]*fvar,
                       label=labels[i], color=colors[i], fmt=fmts[i],
                       capthick=capthick, capsize=capsize,elinewidth=elinewidth)
        ax[1].errorbar(x=t[::step], y=np.mean(rmsd, axis=0)[::step],yerr=0.1*np.var(rmsd, axis=0)[::step]*fvar,
                        color=colors[i], fmt=fmts[i],
                       capthick=capthick, capsize=capsize,elinewidth=elinewidth)
        ax[0].plot(t, np.mean(cc, axis=0), "-",color=colors[i])
        ax[1].plot(t, np.mean(rmsd, axis=0), "-",color=colors[i])
        ax[0].set_xlabel("Simulation Time (ps)")
        ax[0].set_ylabel("CC")
        ax[0].set_title("Correlation Coefficient")
        ax[1].set_xlabel("Simulation Time (ps)")
        ax[1].set_ylabel("RMSD (A)")
        ax[1].set_title("Root Mean Square Deviation")
        # ax[1,0].set_xlim(-1, len(protocol_list))
        # ax[1,0].set_xticks(np.arange(len(protocol_list)))
        # ax[1,0].set_xticklabels(labels)
        # ax[1,0].set_title("ClashScore")
        # ax[1,1].set_xlim(-1, len(protocol_list))
        # ax[1,1].set_xticks(np.arange(len(protocol_list)))
        # ax[1,1].set_xticklabels(labels)
        # ax[1,1].set_title("MolProbityScore")
        fig.tight_layout()
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles = handles, loc='lower right')
    return fig

def show_molprob(protocol_list, length, labels, figsize=(10,5),
                 colors=["tab:blue", "tab:red", "tab:green", "tab:orange",
                         "tab:brown", "tab:olive", "tab:pink", "tab:green", "tab:cyan"]):

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    for i in range(len(protocol_list)):
        molp=[]
        for j in range(length[i]):
            molpfile =  "%s/extra/run_r%i_molprobity.npy" % ( protocol_list[i], j+1)
            if not os.path.exists(molpfile):
                molpfile = "%s/extra/min%i_molprobity.npy" % (protocol_list[i], j + 1)
                if not os.path.exists(molpfile):
                    molpfile = "%s/extra/min_molprobity.npy" % (protocol_list[i])
            molp.append(np.load(molpfile))
            ax[0].plot(i, molp[-1][0], "o", color=colors[i])
            ax[1].plot(i,  molp[-1][1], "o", color=colors[i])
        ax[0].set_xlim(-1, len(protocol_list))
        ax[0].set_xticks(np.arange(len(protocol_list)))
        ax[0].set_xticklabels(labels)
        ax[0].set_title("ClashScore")
        ax[1].set_xlim(-1, len(protocol_list))
        ax[1].set_xticks(np.arange(len(protocol_list)))
        ax[1].set_xticklabels(labels)
        ax[1].set_title("MolProbityScore")

    fig.tight_layout()
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles = handles, loc='lower right')
    return fig

ak = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000754_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003692_FlexProtGenesisFit",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=100, fvar=1, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=51, start=0)

ak_molp = show_molprob([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005266_FlexProtGenesisMin",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005405_FlexProtGenesisMin",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005362_FlexProtGenesisMin",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005309_FlexProtGenesisMin",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000754_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003692_FlexProtGenesisFit",
],
             length=[16,16,1,1, 16,16], labels=["Local", "Global", "Init", "Target", "local2", "global2"], figsize=(10,3))

lao = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003997_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004430_FlexProtGenesisFit",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=100, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(7,7), dt=0.002, end= 51)

lacto = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004647_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005083_FlexProtGenesisFit",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=500, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=51, start=10)


corA = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/CorA/local",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/CorA/global",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=1000, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=51, start=8)

p97 = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/P97/local",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/P97/global",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=1000, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=-1, start=0)

