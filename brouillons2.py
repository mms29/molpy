import matplotlib
# matplotlib.use('Agg')
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
                 fmts = ["o", "d", "v", "^", "p", "*", "s","x"], end=-1, start=0, show_runs=False, nrep=1):
    if labels is None:
        labels = ["#"+str(i) for i in range(len(protocol_list))]
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    for i in range(len(protocol_list)):
        cc = []
        rmsd = []
        molp=[]
        for j in range(start, length[i]):
            f = "%s/extra/run_r%i" %(protocol_list[i],j+1 )
            if not os.path.exists(f+"_cc.npy"):
                f = "%s/extra/run%i_remd%i" %(protocol_list[i],nrep, j+1 )
            if os.path.exists(f + "_cc.npy"):
                cc.append(np.load(f+"_cc.npy")[:end])
                rmsd.append(np.load(f+"_rmsd.npy")[:end])
            else:
                f = "%s/extra/%s_run_remd%i" %(protocol_list[i], str(nrep).zfill(3), j+1 )
                cc.append(np.loadtxt(f + "_cc.txt")[:end])
                rmsd.append(np.loadtxt(f + "_rmsd.txt")[:end])

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
        if not show_runs:
            idx = np.where(rmsd==np.min(rmsd))[0][0]
            print(labels[i]+" best index = "+str(idx))
            ax[0].plot(t, cc[idx], "-", color=colors[i], alpha=0.5)
            ax[1].plot(t, rmsd[idx], "-", color=colors[i], alpha=0.5)
        ax[0].errorbar(x=t[::step], y=np.mean(cc, axis=0)[::step],  yerr=np.std(cc, axis=0)[::step]*fvar,
                       label=labels[i], color=colors[i], fmt=fmts[i],
                       capthick=capthick, capsize=capsize,elinewidth=elinewidth)
        ax[1].errorbar(x=t[::step], y=np.mean(rmsd, axis=0)[::step],yerr=np.std(rmsd, axis=0)[::step]*fvar,
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

def show_cc(protocol_list, length, labels=None, period=100,
                 fvar=10.0, capthick=10.0, capsize=10.0, elinewidth=1.0, figsize=(10,5), dt=0.002,
                 colors=["tab:red", "tab:blue", "tab:green", "tab:orange",
                         "tab:brown", "tab:olive", "tab:pink", "tab:green", "tab:cyan"],
                 fmts = ["o", "d", "v", "^", "p", "*", "s","x"], end=-1, start=0, show_runs=False):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in range(len(protocol_list)):
        cc = []
        for j in range(start, length[i]):
            cc.append(np.load(protocol_list[i]+ "/extra/run_r"+str(j+1)+"_cc.npy")[:end])
            t = np.arange(len(cc[-1])) * period * dt
        step = len(cc[0]) // 10
        ax.errorbar(x=t[::step], y=np.mean(cc, axis=0)[::step],  yerr=np.std(cc, axis=0)[::step]*fvar,
                       label=labels[i], color=colors[i], fmt=fmts[i],
                       capthick=capthick, capsize=capsize,elinewidth=elinewidth)
        ax.plot(t, np.mean(cc, axis=0), "-",color=colors[i])
        ax.set_xlabel("Simulation Time (ps)")
        ax.set_ylabel("CC")
        ax.set_title("Correlation Coefficient")
        fig.tight_layout()
    ax.legend()
    return fig

def show_cc_rmsd_all(N, protocol_list, length, labels=None, period=[100],
                 fvar=10.0, capthick=10.0, capsize=10.0, elinewidth=1.0, figsize=(10,5), dt=0.002,
                 colors=["tab:red", "tab:blue", "tab:green", "tab:orange",
                         "tab:brown", "tab:olive", "tab:pink", "tab:green", "tab:cyan"],
                 fmts = ["o", "d", "v", "^", "p", "*", "s","x"], end=[-1], start=[0], show_runs=False):
    fig, ax = plt.subplots(N, 2, figsize=figsize)
    for n in range(N):
        for i in range(len(protocol_list[n])):
            cc = []
            rmsd = []
            for j in range(start[n], length[n][i]):
                f = "%s/extra/run_r%i" % (protocol_list[n][i], j + 1)
                if not os.path.exists(f + "_cc.npy"):
                    f = "%s/extra/run%i_remd%i" % (protocol_list[n][i], 1, j + 1)
                cc.append(np.load(f + "_cc.npy")[:end[n]])
                rmsd.append(np.load(f + "_rmsd.npy")[:end[n]])
                t = np.arange(len(cc[-1])) * period[n] * dt
                step =  len(cc[-1])//10
            if not show_runs:
                ax[n,0].errorbar(x=t[::step], y=np.mean(cc, axis=0)[::step],  yerr=np.std(cc, axis=0)[::step]*fvar,
                               label=labels[i], color=colors[i], fmt=fmts[i],
                               capthick=capthick, capsize=capsize,elinewidth=elinewidth)
                ax[n,1].errorbar(x=t[::step], y=np.mean(rmsd, axis=0)[::step],yerr=np.std(rmsd, axis=0)[::step]*fvar,
                                color=colors[i], fmt=fmts[i],
                               capthick=capthick, capsize=capsize,elinewidth=elinewidth)
                ax[n,0].plot(t, np.mean(cc, axis=0), "-",color=colors[i])
                ax[n,1].plot(t, np.mean(rmsd, axis=0), "-",color=colors[i])
            else:
                idx = np.where(rmsd == np.min(rmsd))[0][0]
                print(labels[i] + " best index = " + str(idx))
                ax[n,0].plot(t[::step], cc[idx][::step], "-"+fmts[i], color=colors[i])
                ax[n,1].plot(t[::step], rmsd[idx][::step], "-"+fmts[i], color=colors[i])
            ax[n,0].set_ylabel("CC")
            ax[n,1].set_ylabel("RMSD (A)")
    ax[0,0].set_title("Correlation Coefficient")
    ax[0,1].set_title("Root Mean Square Deviation")
    ax[N-1,0].set_xlabel("Simulation Time (ps)")
    ax[N-1,1].set_xlabel("Simulation Time (ps)")
    fig.tight_layout()
    handles, labels = ax[0,0].get_legend_handles_labels()
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
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007162_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012724_FlexProtGenesisFit",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=500, fvar=1, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=-1, start=0,show_runs=False)


ak = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/015339_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/015141_FlexProtGenesisFit",
              ],
             length=[14,14], labels=["local", "global"],
             step=10, period=500, fvar=1, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=-1, start=0,show_runs=False)

lao = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003997_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004430_FlexProtGenesisFit",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=500, fvar=0.5, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end= 21, start=0,show_runs=False)

lacto = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004647_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005083_FlexProtGenesisFit",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=500, fvar=1, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=51, start=3,show_runs=False)


ef2 = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006459_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/013591_FlexProtGenesisFit",
              ],
             length=[16,16], labels=["local", "global"],
             step=10, period=1000, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=200, start=12,
                show_runs=False, nrep=1)

abc = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010519_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010755_FlexProtGenesisFit",
              ],
             length=[16,16], labels=["MD","NMMD"],
             step=10, period=1000, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=-1, start=0, show_runs=True)
abc.savefig("/home/guest/Documents/VirtualBoxShared/pictures/abc_cc_rmsd.png", dpi=1000)
p97 = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber7/run/media/guest/disk4/ScipionUserData/projects/Remi_P97_paper2021/Runs/000306_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber7/run/media/guest/disk4/ScipionUserData/projects/Remi_P97_paper2021/Runs/000376_FlexProtGenesisFit"
              ],
             length=[16,16], labels=["MD", "NMMD"],
              period=1000, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(5,3), dt=0.002, end=201, start=0, show_runs=False)
p97return = show_cc_rmsd([
              "/run/user/1001/gvfs/sftp:host=amber7/run/media/guest/disk4/ScipionUserData/projects/Remi_P97_paper2021/Runs/000306_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber7/run/media/guest/disk4/ScipionUserData/projects/Remi_P97_paper2021/Runs/000672_FlexProtGenesisFit"
              ],
             length=[16,16], labels=["MD", "NMMD"],
              period=1000, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(5,3), dt=0.002, end=201, start=0, show_runs=False)

all = show_cc_rmsd_all(N = 6, protocol_list = [[
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003997_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004430_FlexProtGenesisFit",
              ],[
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/007162_FlexProtGenesisFit",
              "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/012724_FlexProtGenesisFit",
              ],[
    "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/004647_FlexProtGenesisFit",
    "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/005083_FlexProtGenesisFit",
                ],[
    "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/006459_FlexProtGenesisFit",
    "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/013591_FlexProtGenesisFit",
                ],[
    "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010519_FlexProtGenesisFit",
    "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010755_FlexProtGenesisFit",
                ],[
    "/run/user/1001/gvfs/sftp:host=amber7/run/media/guest/disk4/ScipionUserData/projects/Remi_P97_paper2021/Runs/000306_FlexProtGenesisFit",
    "/run/user/1001/gvfs/sftp:host=amber7/run/media/guest/disk4/ScipionUserData/projects/Remi_P97_paper2021/Runs/000376_FlexProtGenesisFit"
                ]],
             length=[[16,16],[16,16],[16,16],[13,13],[16,16],[16,16]], labels=["MD", "NMMD"],
                      period=[500,500,500,1000,1000,1000], fvar=1, capthick=1.7, capsize=5,elinewidth=1.7,
                      figsize=(10,15), dt=0.002, end=[-1,-1,-1,-1,-1,-1], start=[0,0,0,0,0,0],show_runs=False)
all.savefig("/home/guest/Documents/VirtualBoxShared/pictures/PaperFrontiers/all_replicas3.png", dpi=1000)


# corA = show_cc_rmsd([
#               "/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/CorA/51020/local/results20",
#               "/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/CorA/51020/global/results20",
#               ],
#              length=[15,15], labels=["local", "global"],
#              step=10, period=1000, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,15), dt=0.002, end=200, start=0, show_runs=True)
# corA = show_cc_rmsd([
#               "/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/CorA/local3",
#               "/run/user/1001/gvfs/sftp:host=amber9/home/guest/Workspace/PaperFrontiers/CorA/global3",
#               ],
#              length=[16,16], labels=["local", "global"],
#              step=10, period=1000, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=51, start=9, show_runs=False)
# corA201 = show_cc_rmsd([
#               "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/003596_FlexProtGenesisFit",
#               "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009179_FlexProtGenesisFit",
#               ],
#              length=[16,16], labels=["local", "global"],
#              step=10, period=10000, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=-1, start=6, show_runs=True)
# corA20 = show_cc_rmsd([
#               "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009425_FlexProtGenesisFit",
#               "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009499_FlexProtGenesisFit",
#               ],
#              length=[14,14], labels=["MD", "NMMD"], period=1000, fvar=2, capthick=1.7,
#     capsize=5,elinewidth=1.7, figsize=(5,3), dt=0.002, end=-1, start=0, show_runs=False)
# corA20 = show_cc([
#               "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009425_FlexProtGenesisFit",
#               "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009499_FlexProtGenesisFit",
#               ],
#              length=[14,14], labels=["MD", "NMMD"], period=1000, fvar=2, capthick=1.7,
#     capsize=5,elinewidth=1.7, figsize=(5,3), dt=0.002, end=-1, start=10, show_runs=False)
# corA20.savefig("/home/guest/Documents/VirtualBoxShared/pictures/PaperFrontiers/cora_cc.png", dpi=1000)
#
# corA10 = show_cc_rmsd([
#               "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009666_FlexProtGenesisFit",
#               "/run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/009876_FlexProtGenesisFit",
#               ],
#              length=[16,16], labels=["local", "global"],
#              step=10, period=1000, fvar=2, capthick=1.7, capsize=5,elinewidth=1.7, figsize=(10,3), dt=0.002, end=-1, start=0, show_runs=False)

# ChimeraX /run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010519_FlexProtGenesisFit/extra/run_r1.pdb /run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010755_FlexProtGenesisFit/extra/run_r6.pdb /run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010487_FlexProtGeneratePSF/extra/output.pdb /run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010295_FlexProtGeneratePSF/extra/output.pdb /run/user/1001/gvfs/sftp:host=amber9/home/guest/ScipionUserData/projects/PaperFrontiers/Runs/010519_FlexProtGenesisFit/extra/target.sit

plt.plot(np.linspace(0,100),"s", color="tab:blue", label="NMMD")
plt.plot(np.linspace(0,100),"s", color="tab:orange", label="Init.")
plt.plot(np.linspace(0,100),"s", color="tab:grey", label="Target map")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("/home/guest/Documents/VirtualBoxShared/pictures/PaperFrontiers/ga_legend.png", dpi=1000)