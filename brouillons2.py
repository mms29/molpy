
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
                 colors=["tab:red", "tab:blue", "tab:green", "tab:orange",
                         "tab:brown", "tab:olive", "tab:pink", "tab:green", "tab:cyan"],
                 fmts = ["o", "d", "v", "^", "p", "*", "s","x"], end=-1, start=0, show_runs=False):
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
            if show_runs :
                cols = [list(pl.cm.Reds(np.linspace(0.5,1,length[i]))), list(pl.cm.Blues(np.linspace(0.5,1,length[i])))]
                ax[0].plot(t, cc[-1], "-", color=cols[i][j])
                ax[1].plot(t, rmsd[-1], "-", color=cols[i][j])
            # ax[1,0].plot(i, molp[-1][0], "o", color=colors[i])
            # ax[1,1].plot(i, molp[-1][1], "o", color=colors[i])
        ax[0].errorbar(x=t[::step], y=np.mean(cc, axis=0)[::step],  yerr=np.var(cc, axis=0)[::step]*fvar/np.min(cc),
                       label=labels[i], color=colors[i], fmt=fmts[i],
                       capthick=capthick, capsize=capsize,elinewidth=elinewidth)
        ax[1].errorbar(x=t[::step], y=np.mean(rmsd, axis=0)[::step],yerr=np.var(rmsd, axis=0)[::step]*fvar/np.max(rmsd),
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
                 colors=["tab:red", "tab:blue", "tab:green", "tab:orange",
                         "tab:brown", "tab:olive", "tab:pink", "tab:green", "tab:cyan"]):

    fig, ax = plt.subplots(1, 4, figsize=figsize)
    for i in range(len(protocol_list)):
        molp=[]
        for j in range(length[i]):
            if length[i]==1:
                molp.append(np.loadtxt("%s_molprobity.txt" %(protocol_list[i])))
            else:
                molp.append(np.loadtxt("%s%i_molprobity.txt" %(protocol_list[i],j+1)))
        molp = np.array(molp)

        for j in range(4):
            ax[j].boxplot(molp[:, j], positions=[i], notch=False, patch_artist=True,
                        boxprops=dict(facecolor=colors[i], color=colors[i]),
                        capprops=dict(color=colors[i]),
                        whiskerprops=dict(color=colors[i]),
                        flierprops=dict(color=colors[i], markeredgecolor=colors[i]),
                        medianprops=dict(color=colors[i]),
                        )

        ax[0].set_xlim(-1, len(protocol_list))
        ax[0].set_xticks(np.arange(len(protocol_list)))
        ax[0].set_xticklabels(labels[:len(protocol_list)])
        ax[0].set_title("ClashScore")

        ax[1].set_xlim(-1, len(protocol_list))
        ax[1].set_xticks(np.arange(len(protocol_list)))
        ax[1].set_xticklabels(labels[:len(protocol_list)])
        ax[1].set_title("MolProbityScore")

        ax[2].set_xlim(-1, len(protocol_list))
        ax[2].set_xticks(np.arange(len(protocol_list)))
        ax[2].set_xticklabels(labels[:len(protocol_list)])
        ax[2].set_title("ramaFavored")

        ax[3].set_xlim(-1, len(protocol_list))
        ax[3].set_xticks(np.arange(len(protocol_list)))
        ax[3].set_xticklabels(labels[:len(protocol_list)])
        ax[3].set_title("rotaFavored")

    fig.tight_layout()
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles = handles, loc='lower right')
    return fig
ak_molp = show_molprob([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005266_FlexProtGenesisMin/extra/min",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005405_FlexProtGenesisMin/extra/min",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000614_FlexProtGenesisMin/extra/min",
              # "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005309_FlexProtGenesisMin",
              # "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000754_FlexProtGenesisFit",
              # "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003692_FlexProtGenesisFit",
],
             length=[16,16,1,1, 16,16], labels=["Local", "Global", "Init", "Target", "local2", "global2"], figsize=(10,3))

lao_molp = show_molprob([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006162_FlexProtGenesisMin/extra/min",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006200_FlexProtGenesisMin/extra/min",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006974_FlexProtGenesisMin/extra/min",
              # "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005309_FlexProtGenesisMin",
              # "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000754_FlexProtGenesisFit",
              # "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003692_FlexProtGenesisFit",
],
             length=[16,16,1,1, 16,16], labels=["Local", "Global", "Init", "Target", "local2", "global2"], figsize=(10,3))

lacto_molp = show_molprob([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006238_FlexProtGenesisMin/extra/min",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006276_FlexProtGenesisMin/extra/min",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007027_FlexProtGenesisMin/extra/min",
              # "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005309_FlexProtGenesisMin",
              # "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000754_FlexProtGenesisFit",
              # "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003692_FlexProtGenesisFit",
],
             length=[16,16,1,1, 16,16], labels=["Local", "Global", "Init", "Target", "local2", "global2"], figsize=(10,3))

ef2_molp = show_molprob([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006819_FlexProtGenesisMin/extra/min",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006857_FlexProtGenesisMin/extra/min",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006421_FlexProtGenesisMin/extra/min",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006459_FlexProtGenesisFit/extra/run_r",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006745_FlexProtGenesisFit/extra/run_r",
],
             length=[16,16,1,16, 16], labels=["Local", "Global", "Init", "Target", "local2", "global2"], figsize=(10,3))


ak = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/000754_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003692_FlexProtGenesisFit",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=100, fvar=10, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=51, start=0,show_runs=True)
ak = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007088_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007162_FlexProtGenesisFit",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=500, fvar=10, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=-1, start=0,show_runs=True)

lao = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003997_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004430_FlexProtGenesisFit",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=500, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end= 51, start=0,show_runs=False)

lacto = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004647_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005083_FlexProtGenesisFit",
              ],
             length=[15,15], labels=["local", "global"],
             step=10, period=500, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=51, start=4,show_runs=False)

ef2 = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006459_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006745_FlexProtGenesisFit",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=1000, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=200, start=0,
                show_runs=True)

corA = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/CorA/local",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/CorA/global",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=1000, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=51, start=0, show_runs=True)

p97 = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/P97/local",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/P97/global",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=1000, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=-1, start=0, show_runs=True)

