import autograd.numpy as npg
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_RMSD(psim, pexp):
    return np.linalg.norm(psim-pexp)**2

def select_voxels(coord, size, sampling_rate, threshold):
    n_atoms = coord.shape[0]
    n_vox = threshold*2 +1
    l=np.zeros((n_atoms,3))

    for i in range(n_atoms):
        l[i] = (coord[i]/sampling_rate -threshold + size/2).astype(int)

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

def cross_correlation(map1, map2):
    return np.sum(map1*map2)/np.sqrt(np.sum(np.square(map1))*np.sum(np.square(map2)))

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


def compute_pca(data, length, labels=None, save=None, n_components=2):
    print("Computing PCA ...")

    # Compute PCA
    arr = np.array(data)
    pca = PCA(n_components=n_components)
    pca.fit(arr.T)

    # Prepare plotting data
    idx = np.concatenate((np.array([0]),np.cumsum(length))).astype(int)
    if labels is None:
        labels = ["#"+str(i) for i in range(len(length))]

    # Plotter
    fig = plt.figure()
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel("PCA component 3")
    else:
        ax = fig.add_subplot(111)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    for i in range(len(length)):
        if n_components==3:
            ax.scatter(pca.components_[0, idx[i]:idx[i+1]], pca.components_[1, idx[i]:idx[i+1]],
                       pca.components_[2, idx[i]:idx[i+1]], s=10, label=labels[i])
        else:
            ax.plot(pca.components_[0, idx[i]:idx[i+1]], pca.components_[1, idx[i]:idx[i+1]], 'o',
                    label=labels[i], markeredgecolor='black')
    ax.legend()
    if save is not None:
        fig.savefig(save)
    print("Done")


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


def pdb2vol(coord, size, sigma, voxel_size, threshold):
    vox, n_vox = select_voxels(coord, size, voxel_size, threshold)
    n_atoms = coord.shape[0]
    vol = np.zeros((size, size, size))
    expnt = np.zeros((n_atoms, n_vox, n_vox, n_vox))
    for i in range(n_atoms):
        mu = (np.mgrid[vox[i, 0]:vox[i, 0] + n_vox,
              vox[i, 1]:vox[i, 1] + n_vox,
              vox[i, 2]:vox[i, 2] + n_vox] - size / 2) * voxel_size
        x = np.repeat(coord[i], n_vox ** 3).reshape(3, n_vox, n_vox, n_vox)
        expnt[i] = np.exp(-np.square(np.linalg.norm(x - mu, axis=0)) / (2 * (sigma ** 2)))
        vol[vox[i, 0]:vox[i, 0] + n_vox,
        vox[i, 1]:vox[i, 1] + n_vox,
        vox[i, 2]:vox[i, 2] + n_vox] += expnt[i]
    return vol, expnt

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

