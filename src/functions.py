import autograd.numpy as npg
import numpy as np

def get_RMSD(psim, pexp):
    return np.linalg.norm(psim-pexp)**2

def select_voxels(coord, size, sampling_rate, threshold):
    n_atoms = coord.shape[0]
    n_vox = threshold*2 +1
    l=np.zeros((n_atoms,3))

    for i in range(n_atoms):
        l[i] = (coord[i]/sampling_rate -threshold + size/2).astype(int)

    if (np.max(l) >= size or np.min(l)<0):
        raise RuntimeError("threshold too large")
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


