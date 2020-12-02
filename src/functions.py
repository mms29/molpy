import numpy as np
import pystan
import pickle
import os.path
import hashlib

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def read_pdb(file, pattern='CA'):
    x = []
    ca=[]
    with open(file, "r") as file :
        end=False
        for line in file:
            l = line.split()
            if len(l) >0:
                if l[0] == 'ATOM':
                    if not end:
                        x.append([float(l[6]), float(l[7]), float(l[8])])
                        if l[2] == pattern:
                            ca.append(len(x))
                if l[0] == 'TER':
                    end = True

    x = np.array(x)
    return x, ca

def read_modes(file, n_modes=20, skip_first_modes=True):
    A = []
    if skip_first_modes: fm=6
    else: fm=0
    for i in range(fm, n_modes+fm):
        A.append(np.loadtxt(file + str(i + 1)))
    return np.transpose(np.array(A),(1,0,2))

def read_stan_model(model, save=True, build=False, threads=False):
    if not os.path.exists('stan/saved_stan_models/'+model+'.pkl'):
        build=True
    if not os.path.exists('stan/saved_stan_models/'+model+'.checksum'):
        build=True
    else:
        checksum = md5('stan/'+model+'.stan')
        with open('stan/saved_stan_models/'+model+'.checksum', 'r') as f:
            old_checksum =f.read()
        if checksum != old_checksum:
            build=True
    if build==True:
        s=""
        with open('stan/'+model+'.stan', "r") as f:
            for line in f:
                s+=line
        if threads:
            extra_compile_args = ['-pthread', '-DSTAN_THREADS']
            sm = pystan.StanModel(model_code=s, extra_compile_args=extra_compile_args)
        else :
            sm = pystan.StanModel(model_code=s)
        if save==True:
            with open('stan/saved_stan_models/'+model+'.pkl', 'wb') as f:
                pickle.dump(sm, f)
            with open('stan/saved_stan_models/'+model+'.checksum', 'w') as f:
                f.write(md5('stan/'+model+'.stan'))
    else:
        sm = pickle.load(open('stan/saved_stan_models/'+model+'.pkl', 'rb'))
    return sm

def rotate_pdb(self, atoms, angles):
    a,b,c =angles
    cos = np.cos
    sin=np.sin
    R = [[cos(a) * cos(b), cos(a) * sin(b) * sin(c) - sin(a) * cos(c), cos(a) * sin(b) * cos(c) + sin(a) * sin(c)],
         [sin(a) * cos(b), sin(a) * sin(b) * sin(c) + cos(a) * cos(c), sin(a) * sin(b) * cos(c) - cos(a) * sin(c)],
         [-sin(b), cos(b) * sin(c), cos(b) * cos(c)]];
    rotated_atoms = np.zeros(atoms.shape)
    for i in range(atoms.shape[0]):
        rotated_atoms[i] = atoms[i]*R
    return rotated_atoms


def volume_from_pdb(x, N, sigma=1, sampling_rate=1, precision=0.001):

    halfN=int(N/2)
    if((x < -N*sampling_rate/2).any() or (x > N*sampling_rate/2).any()):
        raise RuntimeError("WARNING !! box size = -"+str(np.max([
            (N* sampling_rate / 2)  - np.max(x),
            (N * sampling_rate / 2)  + np.min(x)]
        )))
    else:
        print("box size = "+str(np.max([
            (N* sampling_rate / 2)  - np.max(x),
            (N * sampling_rate / 2)  + np.min(x)]
        )))
    n_atoms = x.shape[0]
    em_density = np.zeros((N,N,N))

    guassian_range = 0
    while(gaussian_pdf(np.array([guassian_range*sampling_rate,0,0]),np.array([0,0,0]), sigma) > precision):
        guassian_range+=1


    for a in range(n_atoms):
        pos = np.rint((x[a]/sampling_rate) + halfN).astype(int)
        for i in range(pos[0] - guassian_range , pos[0] + guassian_range + 1):
            if (i>=0 and i <N):
                for j in range(pos[1] - guassian_range , pos[1] + guassian_range + 1):
                    if (j >= 0 and j < N):
                        for k in range(pos[2] - guassian_range, pos[2] +guassian_range + 1):
                            if (k >= 0 and k < N):
                                em_density[i, j, k] += gaussian_pdf(np.array([i-halfN,j-halfN,k-halfN])*sampling_rate,x[a], sigma)

    return em_density


def gaussian_pdf(x, mu, sigma):
    return np.exp(-((x[0]-mu[0])**2 + (x[1]-mu[1])**2 + (x[2]-mu[2])**2)/(2*(sigma**2)))

def volume_from_pdb_slow(x, size, sigma, sampling_rate=1):
    n_atoms = x.shape[0]
    em_density = np.zeros(size)
    center_transform = np.array([size[0] / 2, size[1] / 2, size[2] / 2]).astype(int)
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                for a in range(n_atoms):
                    em_density[i, j, k] += gaussian_pdf(x[a],((np.array([i, j, k]) - center_transform) * sampling_rate) , sigma)
    return em_density

def center_pdb(x):
    return x - np.mean(x, axis=0)

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



def md_bonds_potential(x, k_r, r0):
    r = np.linalg.norm(x[1:] - x[:-1], axis=1)
    return np.sum( k_r * np.square(r - r0))

def md_angles_potential(x, k_theta, theta0):
    theta = np.arccos( np.sum( (x[:-2]-x[1:-1]) * (x[1:-1]-x[2:]), axis=1)
                      / (np.linalg.norm(x[:-2] - x[1:-1], axis=1) * np.linalg.norm(x[1:-1] - x[2:], axis=1)))
    return np.sum(k_theta * np.square(theta - theta0))

def md_lennard_jones_potential(x, k_lj, d_lj):
    U = 0.0
    n_atoms = x.shape[0]
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                U += 4 * k_lj * (
                            (d_lj / np.linalg.norm(x[i] - x[j])) ** 12 - (d_lj / np.linalg.norm(x[i] - x[j])) ** 6)
    return U

def md_lennard_jones_potential(x, k_lj, d_lj):
    n_atoms = x.shape[0]
    dist = np.linalg.norm(np.repeat(x, n_atoms, axis=0).reshape(n_atoms, n_atoms, 3) - x, axis=2)
    np.fill_diagonal(dist, np.nan)
    inv = d_lj/dist
    return np.nansum(4 * k_lj * (inv**12 - inv**6))

#
#
# def md_bonds_potential(x, k_r, r0):
#     U=0.0
#     n_atoms = x.shape[0]
#     for i in range(n_atoms - 1):
#         r = np.linalg.norm(x[i] - x[i + 1])
#         U += k_r * (r - r0) ** 2
#     return U;
#
# def md_angles_potential(x, k_theta, theta0):
#     U = 0.0
#     n_atoms = x.shape[0]
#
#     for i in range(n_atoms - 2):
#         theta = np.arccos(np.dot(x[i] - x[i + 1], x[i + 1] - x[i + 2])
#                           / (np.linalg.norm(x[i] - x[i + 1]) * np.linalg.norm(x[i + 1] - x[i + 2])))
#         U += k_theta * (theta - theta0) ** 2
#     return U
#
# def md_lennard_jones_potential(x, k_lj, d_lj):
#     U = 0.0
#     n_atoms = x.shape[0]
#     for i in range(n_atoms):
#         for j in range(n_atoms):
#             if i != j:
#                 U += 4 * k_lj * (
#                             (d_lj / np.linalg.norm(x[i] - x[j])) ** 12 - (d_lj / np.linalg.norm(x[i] - x[j])) ** 6)
#     return U