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

    # Compute PCA
    arr = np.array(data)
    pca = PCA(n_components=n_components)
    pca.fit(arr.T)

    # Prepare plotting data
    idx = np.concatenate((np.array([0]),np.cumsum(length))).astype(int)
    print(idx)
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
