# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
# * IMPMC, UPMC Sorbonne University
# *
# **************************************************************************

import autograd.numpy as npg
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import contextlib
import io
import sys
# import seaborn as sns

def select_voxels(coord, size, voxel_size, cutoff):
    n_atoms = coord.shape[0]
    threshold = int(np.ceil(cutoff/voxel_size))
    n_vox = int(threshold*2 +1)
    l=np.zeros((n_atoms,3))
    for i in range(n_atoms):
        l[i] = (coord[i]/voxel_size -threshold + size/2).astype(int)

    if (np.max(l) >= size or np.min(l)<0) or (np.max(l+n_vox) >= size or np.min(l+n_vox)<0):
        raise RuntimeError("ERROR : Atomic coordinates got outside the box")
    return l.astype(int), n_vox

def to_vector(arr):
    X,Y,Z = arr.shape
    vec = np.zeros(X*Y*Z)
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                vec[z + y*Z + x*Y*Z] = arr[x,y,z]
    return vec

def to_matrix(vec, X,Y,Z):
    arr = np.zeros((X,Y,Z))
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                arr[x, y, z] = vec[z + y * Z + x * Y * Z]
    return arr

def generate_euler_matrix(angles):
    a, b, c = angles
    cos = npg.cos
    sin = npg.sin
    R = npg.array([[ cos(c) *  cos(b) * cos(a) -  sin(c) * sin(a), cos(c) * cos(b) * sin(a) +  sin(c) * cos(a), -cos(c) * sin(b)],
                  [- sin(c) *  cos(b) * cos(a) - cos(c) * sin(a), - sin(c) * cos(b) * sin(a) + cos(c) * cos(a), sin(c) * sin(b)],
                  [sin(b) * cos(a), sin(b) * sin(a), cos(b)]])
    return R

def get_euler_grad(angles, coord):
    a,b,c = angles
    x, y, z = coord
    cos = np.cos
    sin = np.sin

    dR = np.array([[x* (cos(c) *  cos(b) * -sin(a) -  sin(c) * cos(a)) + y* ( cos(c) * cos(b) * cos(a) +  sin(c) * -sin(a)),
                    x* (- sin(c) *  cos(b) * -sin(a) - cos(c) * cos(a)) + y* ( - sin(c) * cos(b) * cos(a) + cos(c) * -sin(a)),
                    x* (sin(b) * -sin(a)) + y* (sin(b) * cos(a))],

                   [x* (cos(c) * -sin(b) * cos(a)) + y* ( cos(c) * -sin(b) * sin(a)) + z* ( -cos(c) * cos(b)),
                    x* (- sin(c) *  -sin(b) * cos(a)) + y* ( - sin(c) * -sin(b) * sin(a)) + z* (sin(c) * cos(b)),
                    x* (cos(b) * cos(a)) + y* (cos(b) * sin(a)) + z* (-sin(b))],

                   [x* (-sin(c) *  cos(b) * cos(a) -  cos(c) * sin(a)) + y* ( -sin(c) * cos(b) * sin(a) +  cos(c) * cos(a)) + z* ( sin(c) * sin(b)),
                    x * (- cos(c) * cos(b) * cos(a) + sin(c) * sin(a)) + y* ( - cos(c) * cos(b) * sin(a) - sin(c) * cos(a))+ z* (cos(c) * sin(b)),
                    0]])

    return dR
def compute_pca(data, length, labels=None, n_components=2, figsize=(5,5)):
    print("Computing PCA ...")
    colors = ["tab:red", "tab:blue", "tab:orange", "tab:green",
              "tab:brown", "tab:olive", "tab:pink", "tab:green", "tab:cyan"]
    # Compute PCA
    arr = np.array(data)
    print(arr.shape)
    pca = PCA(n_components=n_components)
    pca.fit(arr.T)

    # Prepare plotting data
    idx = np.concatenate((np.array([0]),np.cumsum(length))).astype(int)
    if labels is None:
        labels = ["#"+str(i) for i in range(len(length))]

    fig = plt.figure(figsize=figsize)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel("PCA component 3")
    else:
        ax = fig.add_subplot(111)

    for i in range(len(length)):
        if n_components==3:
            ax.scatter(pca.components_[0, idx[i]:idx[i+1]], pca.components_[1, idx[i]:idx[i+1]],
                       pca.components_[2, idx[i]:idx[i+1]], s=10, label=labels[i], color = colors[i])
        else:
            ax.plot(pca.components_[0, idx[i]:idx[i+1]], pca.components_[1, idx[i]:idx[i+1]], 'o',
                    label=labels[i], markeredgecolor='black', color = colors[i])
    ax.legend()
    fig.tight_layout()
    return fig

def compute_pca_pf(data, length, labels=None, save=None, n_components=2, figsize=(5,5), lim=None, lim2=None, lim3=None):
    print("Computing PCA ...")
    colors = ["tab:red", "tab:blue", "tab:orange", "tab:green",
              "tab:brown", "tab:olive", "tab:pink", "tab:green", "tab:cyan"]
    # Compute PCA
    arr = np.array(data)
    print(arr.shape)
    pca = PCA(n_components=n_components)
    # pca.fit(arr.T)
    # components = (pca.components_.T* pca.explained_variance_).T
    components = pca.fit_transform(arr).T

    # Prepare plotting data
    idx = np.concatenate((np.array([0]),np.cumsum(length))).astype(int)
    if labels is None:
        labels = ["#"+str(i) for i in range(len(length))]

    # Plotter
    # fig = plt.figure(figsize=figsize)
    # if n_components == 3:
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.set_zlabel("PCA component 3")
    # else:
    #     ax = fig.add_subplot(111)

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot2grid((3, 4), (0, 1), colspan=3, rowspan=3, fig=fig)
    ax2 = plt.subplot2grid((3, 4), (0, 0), colspan=1, rowspan=3, fig=fig)
    ax.set_xlabel("PCA component 2")
    ax.set_yticklabels([])
    ax2.set_ylabel("PCA component 1")

    for i in range(len(length)):
        if n_components==3:
            ax.scatter(components[0, idx[i]:idx[i+1]], components[1, idx[i]:idx[i+1]],
                       components[2, idx[i]:idx[i+1]], s=10, label=labels[i], color = colors[i])
        elif n_components==2:
            ax.plot(components[1, idx[i]:idx[i+1]], components[0, idx[i]:idx[i+1]], 'o',
                    label=labels[i], markeredgecolor='black', color = colors[i])
            x = components[1, idx[i]:idx[i+1]]
            y = components[0, idx[i]:idx[i+1]]
            if len(x)>1:
                sns.kdeplot(x, y=y, ax=ax, color=colors[i], alpha=0.5)

    ax.legend()
    if save is not None:
        fig.savefig(save)
    print("Done")

    for i in range(len(length)):
        x = components[0, idx[i]:idx[i + 1]]
        if len (x) == 1:
            ax2.axhline(x, color=colors[i])
        else:
           sns.kdeplot(y=x, color=colors[i], alpha=0.5,fill=True,ax=ax2)
    ax2.invert_xaxis()
    ax2.set_xlabel("")
    ax2.set_xticklabels([])
    ax2.set_xticks([])
    if lim is not None:
        ax.set_xlim(lim[0], lim[1])
    if lim2 is not None:
        ax.set_ylim(lim2[0], lim2[1])
        ax2.set_ylim(lim2[0], lim2[1])
    if lim3 is not None:
        ax2.set_xlim(lim3[0], lim3[1])
    fig.tight_layout()
    return fig, pca


def get_RMSD_coords(coords1,coords2):
    return np.sqrt(np.mean(np.square(np.linalg.norm(coords1-coords2, axis=1))))


def show_rmsd_fit(mol, fit, save=None):
    if isinstance(fit.fit, list):
        fits = fit.fit
    else:
        fits = [fit.fit]
    fig, ax = plt.subplots(1,1)
    for i in range(len(fits)):
        rmsd = [get_RMSD_coords(n, mol.coords) for n in fits[i]["coord"]]
        ax.plot(rmsd)
    ax.set_ylabel("RMSD (A)")
    ax.set_xlabel("HMC iteration")
    ax.set_title("RMSD")
    if save is not None:
        fit.savefig(save)

def select_idx(param, idx):
    new_param = []
    new_param_idx = []
    (n_param, len_param) = param.shape
    for i in range(n_param):
        elem = np.zeros(len_param)
        bool = True
        for j in range(len_param):
            if param[i, j] in idx:
                elem[j] = np.where(idx == param[i, j])[0][0]
            else:
                bool = False
                break
        if bool:
            new_param.append(elem)
            new_param_idx.append(i)
    new_param = np.array(new_param).astype(int)
    new_param_idx = np.array(new_param_idx).astype(int)
    return new_param, new_param_idx

def get_mol_conv(mol1,mol2, ca_only=False):
    print("> Converting molecule coordinates ...")
    id1 = []
    id2 = []
    id1_idx = []
    id2_idx = []

    if mol1.chainName[0] in mol2.chainName:
        for i in range(mol1.n_atoms):
            if (not ca_only) or mol1.atomName[i] == "CA":
                id1.append(mol1.chainName[i] + str(mol1.resNum[i]) + mol1.atomName[i])
                id1_idx.append(i)
        for i in range(mol2.n_atoms):
            if (not ca_only) or mol2.atomName[i] == "CA":
                id2.append(mol2.chainName[i] + str(mol2.resNum[i]) + mol2.atomName[i])
                id2_idx.append(i)
    elif mol1.chainID[0] in mol2.chainID:
        for i in range(mol1.n_atoms):
            if (not ca_only) or mol1.atomName[i] == "CA":
                id1.append(mol1.chainID[i] + str(mol1.resNum[i]) + mol1.atomName[i])
                id1_idx.append(i)
        for i in range(mol2.n_atoms):
            if (not ca_only) or mol2.atomName[i] == "CA":
                id2.append(mol2.chainID[i] + str(mol2.resNum[i]) + mol2.atomName[i])
                id2_idx.append(i)
    else:
        print("\t Warning : No matching coordinates")
    id1 = np.array(id1)
    id2 = np.array(id2)
    id1_idx = np.array(id1_idx)
    id2_idx = np.array(id2_idx)

    idx = []
    for i in range(len(id1)):
        idx_tmp = np.where(id1[i] == id2)[0]
        if len(idx_tmp) == 1:
            idx.append([id1_idx[i], id2_idx[idx_tmp[0]]])

    if len(idx)==0:
        print("\t Warning : No matching coordinates")
    print("\t Done")

    return np.array(idx)

def get_mols_conv(mols):
    print("> Converting molecule coordinates ...")
    n_mols = len(mols)

    if mols[0].chainName[0] in mols[1].chainName:
        chaintype = 0
    elif mols[0].chainID[0] in mols[1].chainID:
        chaintype = 1
    else:
        raise RuntimeError("\t Warning : No matching chains")

    ids = []
    ids_idx = []
    for m in mols :
        id_tmp=[]
        id_idx_tmp=[]
        for i in range(m.n_atoms):
                id_tmp.append(m.chainName[i] + str(m.resNum[i]) + m.atomName[i])
                id_idx_tmp.append(i)
        ids.append(np.array(id_tmp))
        ids_idx.append(np.array(id_idx_tmp))

    idx = []
    for i in range(len(ids[0])):
        idx_line = [ids_idx[0][i]]
        for m in range(1,n_mols):
            idx_tmp = np.where(ids[0][i] == ids[m])[0]
            if len(idx_tmp) == 1:
                idx_line.append(ids_idx[m][idx_tmp[0]])
        if len(idx_line) == n_mols :
            idx.append(idx_line)

    if len(idx)==0:
        print("\t Warning : No matching coordinates")
    print("\t Done")

    return np.array(idx)

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

def get_cc_rmsd(N, prefix, target, size, voxel_size, cutoff, sigma, step=1, test_idx=False):
    from src.density import Volume, get_CC
    from src.molecule import Molecule
    target.center()
    target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=voxel_size, cutoff=cutoff, sigma=sigma)
    rmsd=[]
    cc =[]
    idx = None
    for i in range(0,N,step):
        print(i)
        mol = Molecule(prefix+str(i+1)+".pdb")
        mol.center()

        if test_idx:
            if idx is None:
                idx = get_mol_conv(mol, target)
            if len(idx)>0 :
                rmsd.append(get_RMSD_coords(mol.coords[idx[:,0]], target.coords[idx[:,1]]))
            else:
                rmsd.append(0)
        else:
            rmsd.append(get_RMSD_coords(mol.coords, target.coords))
        vol = Volume.from_coords(coord=mol.coords, size=size, voxel_size=voxel_size, cutoff=cutoff, sigma=sigma)
        cc.append(get_CC(vol.data,target_density.data))
        np.save(file=prefix+"cc.npy", arr=np.array(cc))
        np.save(file=prefix+"rmsd.npy", arr=np.array(rmsd))
    return np.array(cc), np.array(rmsd)

