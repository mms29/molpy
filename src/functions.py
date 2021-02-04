import numpy as np
import pystan
import pickle
import os.path
import hashlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        n=0
        for line in file:
            l = line.split()
            if len(l) >0:
                if l[0] == 'ATOM':
                    if not end:
                        x.append([float(l[6]), float(l[7]), float(l[8])])
                        if l[2] == pattern:
                            ca.append(n)
                        n += 1

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

def rotate_pdb(atoms, angles):
    a,b,c =angles
    cos = np.cos
    sin=np.sin
    R = [[cos(a) * cos(b), cos(a) * sin(b) * sin(c) - sin(a) * cos(c), cos(a) * sin(b) * cos(c) + sin(a) * sin(c)],
         [sin(a) * cos(b), sin(a) * sin(b) * sin(c) + cos(a) * cos(c), sin(a) * sin(b) * cos(c) - cos(a) * sin(c)],
         [-sin(b), cos(b) * sin(c), cos(b) * cos(c)]];
    rotated_atoms = np.zeros(atoms.shape)
    for i in range(atoms.shape[0]):
        rotated_atoms[i] = np.dot(atoms[i], R)
    return rotated_atoms

def volume_from_pdb_fast(coord, size, sigma, sampling_rate=1):
    mu= (np.mgrid[0:size, 0:size, 0:size] - size/2)*sampling_rate
    x =np.repeat(coord, size**3).reshape(coord.shape[0], 3, size, size, size)
    return np.sum(np.exp(-np.square(np.linalg.norm(x - mu, axis=1))/(2*(sigma ** 2))), axis=0)

def volume_from_pdb_fast2(coord, size, sigma, sampling_rate=1):
    mu= (np.mgrid[0:size, 0:size, 0:size] - size/2)*sampling_rate
    n_atoms= coord.shape[0]
    vol = np.zeros((size, size, size))
    for i in range(n_atoms):
        x = np.repeat(coord[i], size ** 3).reshape(3, size, size, size)
        vol+=np.exp(-np.square(np.linalg.norm(x - mu, axis=0))/(2*(sigma ** 2)))

    return vol

def volume_from_pdb_fast3(coord, size, sigma, sampling_rate=1, threshold=2):
    vox, n_vox = select_voxels(coord, size, sampling_rate, threshold)
    n_atoms= coord.shape[0]
    vol = np.zeros((size, size, size))
    for i in range(n_atoms):
        mu = (np.mgrid[vox[i,0]:vox[i,0]+n_vox,
                         vox[i,1]:vox[i,1]+n_vox,
                         vox[i,2]:vox[i,2]+n_vox] - size / 2) * sampling_rate
        x = np.repeat(coord[i], n_vox ** 3).reshape(3, n_vox, n_vox, n_vox)
        vol[vox[i,0]:vox[i,0]+n_vox,
            vox[i,1]:vox[i,1]+n_vox,
            vox[i,2]:vox[i,2]+n_vox]+=np.exp(-np.square(np.linalg.norm(x - mu, axis=0))/(2*(sigma ** 2)))

    return vol

def get_RMSD(psim, pexp):
    return np.linalg.norm(psim-pexp)**2

def get_grad_RMSD(coord, psim, pexp, size, sampling_rate, sigma):

    n_atoms = coord.shape[0]
    pdiff = psim - pexp

    dx = np.zeros(coord.shape)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                mu = ((np.array([i, j, k]) - np.ones(3) * (size / 2)) * sampling_rate)
                dpsim =-(1/(sigma**2)) * (coord-mu) * np.repeat(np.exp(-np.square(np.linalg.norm(coord-mu, axis=1))/(2*(sigma ** 2))),3).reshape(n_atoms,3)
                dx += 2* pdiff[i,j,k]*dpsim
    return dx

def get_grad_RMSD2(coord, psim, pexp, size, sampling_rate, sigma):

    n_atoms = coord.shape[0]
    pdiff = psim - pexp

    dx = np.zeros(coord.shape)
    mu = (np.mgrid[0:size, 0:size, 0:size] - size / 2) * sampling_rate
    for i in range(n_atoms):
        x = np.repeat(coord[i], size ** 3).reshape(3, size, size, size)
        tmp = 2* pdiff*np.exp(-np.square(np.linalg.norm(x-mu, axis=0))/(2*(sigma ** 2)))
        dpsim =-(1/(sigma**2)) * (x-mu) * np.array([tmp,tmp,tmp])
        dx[i] = np.sum(dpsim, axis=(1,2,3))

    return dx


def get_grad_RMSD3(coord, psim, pexp, size, sampling_rate, sigma, threshold):
    vox, n_vox = select_voxels(coord, size, sampling_rate, threshold)
    n_atoms = coord.shape[0]
    pdiff = psim - pexp

    dx = np.zeros(coord.shape)
    for i in range(n_atoms):
        mu = (np.mgrid[vox[i, 0]:vox[i, 0] + n_vox,
              vox[i, 1]:vox[i, 1] + n_vox,
              vox[i, 2]:vox[i, 2] + n_vox] - size / 2) * sampling_rate
        x = np.repeat(coord[i], n_vox ** 3).reshape(3, n_vox, n_vox, n_vox)
        tmp = 2* pdiff[vox[i,0]:vox[i,0]+n_vox,
            vox[i,1]:vox[i,1]+n_vox,
            vox[i,2]:vox[i,2]+n_vox]*np.exp(-np.square(np.linalg.norm(x-mu, axis=0))/(2*(sigma ** 2)))
        dpsim =-(1/(sigma**2)) * (x-mu) * np.array([tmp,tmp,tmp])
        dx[i] = np.sum(dpsim, axis=(1,2,3))

    return dx

def get_grad_RMSD_NMA(coord, psim, pexp, size, sampling_rate, sigma, A_modes):
    # NB : coord = x0 + q* A

    n_atoms = coord.shape[0]
    n_modes = A_modes.shape[1]
    pdiff = psim - pexp

    dx = np.zeros(coord.shape)
    dq = np.zeros(n_modes)
    mu = (np.mgrid[0:size, 0:size, 0:size] - size / 2) * sampling_rate
    for i in range(n_atoms):
        x = np.repeat(coord[i], size ** 3).reshape(3, size, size, size)
        tmp = 2* pdiff*np.exp(-np.square(np.linalg.norm(x-mu, axis=0))/(2*(sigma ** 2)))

        dpsimx =-(1/(sigma**2)) * (x-mu) * np.array([tmp,tmp,tmp])
        dx[i] = np.sum(dpsimx, axis=(1,2,3))

        dpsimq = -(1 / (sigma ** 2)) * (x - mu) * np.array([tmp, tmp, tmp])
        dq += np.dot(A_modes[i] , np.sum(dpsimq, axis=(1, 2, 3)))

    return dx, dq

def get_grad_RMSD3_NMA(coord, psim, pexp, size, sampling_rate, sigma, A_modes, threshold):
    # NB : coord = x0 + q* A
    vox, n_vox = select_voxels(coord, size, sampling_rate, threshold)
    n_atoms = coord.shape[0]
    n_modes = A_modes.shape[1]
    pdiff = psim - pexp

    dx = np.zeros(coord.shape)
    dq = np.zeros(n_modes)
    for i in range(n_atoms):
        mu = (np.mgrid[vox[i, 0]:vox[i, 0] + n_vox,
              vox[i, 1]:vox[i, 1] + n_vox,
              vox[i, 2]:vox[i, 2] + n_vox] - size / 2) * sampling_rate
        x = np.repeat(coord[i], n_vox ** 3).reshape(3, n_vox, n_vox, n_vox)
        tmp = 2 * pdiff[vox[i, 0]:vox[i, 0] + n_vox,
                  vox[i, 1]:vox[i, 1] + n_vox,
                  vox[i, 2]:vox[i, 2] + n_vox] * np.exp(-np.square(np.linalg.norm(x - mu, axis=0)) / (2 * (sigma ** 2)))

        dpsimx =-(1/(sigma**2)) * (x-mu) * np.array([tmp,tmp,tmp])
        dx[i] = np.sum(dpsimx, axis=(1,2,3))

        dpsimq = -(1 / (sigma ** 2)) * (x - mu) * np.array([tmp, tmp, tmp])
        dq += np.dot(A_modes[i] , np.sum(dpsimq, axis=(1, 2, 3)))

    return dx, dq

def volume_from_pdb(x, N, sigma, sampling_rate=1):
    halfN = int(N / 2)
    if ((x < -N * sampling_rate / 2).any() or (x > N * sampling_rate / 2).any()):
        print("WARNING !! box size = -" + str(np.max([
            (N * sampling_rate / 2) - np.max(x),
            (N * sampling_rate / 2) + np.min(x)]
        )))
    else:
        print("box size = " + str(np.max([
            (N * sampling_rate / 2) - np.max(x),
            (N * sampling_rate / 2) + np.min(x)]
        )))
    n_atoms = x.shape[0]
    em_density = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                mu = ((np.array([i, j, k]) - np.ones(3)*(N/2)) * sampling_rate)
                em_density[i, j, k] = np.sum(np.exp(-np.square(np.linalg.norm(x-mu, axis=1))/(2*(sigma ** 2))))
    return em_density

def select_voxels(coord, size, sampling_rate, threshold):
    n_atoms = coord.shape[0]
    n_vox = threshold*2 +1
    vox_range = np.array([np.arange(n_vox),
                          np.arange(n_vox),
                          np.arange(n_vox)])
    l=np.zeros((n_atoms,3))

    for i in range(n_atoms):
        l[i] = (coord[i]/sampling_rate -threshold + size/2).astype(int)

    if (np.max(l) >= size or np.min(l)<0):
        raise RuntimeError("threshold too large")
    return l.astype(int), n_vox


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
    n_atoms = x.shape[0]
    dist = np.linalg.norm(np.repeat(x, n_atoms, axis=0).reshape(n_atoms, n_atoms, 3) - x, axis=2)
    np.fill_diagonal(dist, np.nan)
    inv = d_lj/dist
    return np.nansum(4 * k_lj * (inv**12 - inv**6))

#
# #
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

def cross_correlation(map1, map2):
    return np.sum(map1*map2)/np.sqrt(np.sum(np.square(map1))*np.sum(np.square(map2)))

def root_mean_square_error(map1, map2):
    return np.sqrt(np.mean(np.square(map1- map2)))/np.max([np.max(map1)-np.min(map1), np.max(map2)-np.min(map2)])

def compute_u_init(structure, bonds=None, angles=None, lennard_jones=None):
    U_init=0
    n_atoms=structure.shape[0]
    if bonds is not None:
        bonds_r0= np.mean([np.linalg.norm(structure[i] - structure[i + 1]) for i in range(n_atoms-1)])
        bonds_k = bonds['k']
        bonds= md_bonds_potential(structure, bonds_k ,bonds_r0)
        U_init+=bonds
        print('Bonds')
        print('\t|--- k='+str(bonds_k))
        print('\t|--- r0='+str(bonds_r0))
        print('\t|--- U='+str(bonds))
    if angles is not None:
        theta0 = []
        for i in range(n_atoms - 2):
            theta0.append(np.arccos(np.dot(structure[i] - structure[i + 1], structure[i + 1] - structure[i + 2])
                                       / (np.linalg.norm(structure[i] -structure[i + 1]) * np.linalg.norm(structure[i + 1] - structure[i + 2]))))
        angles_theta0 = np.mean(theta0)
        angles_k= angles['k']
        angles= md_angles_potential(structure, angles_k, angles_theta0)
        U_init += angles
        print('Angles')
        print('\t|--- k='+str(angles_k))
        print('\t|--- t0='+str(angles_theta0))
        print('\t|--- U='+str(angles))
    if lennard_jones is not None:
        lennard_jones_k = lennard_jones ['k']
        lennard_jones_d =  lennard_jones ['d']
        lj = md_lennard_jones_potential(structure, lennard_jones_k, lennard_jones_d)
        print('Lennard-Jones')
        print('\t|--- k=' + str(lennard_jones_k))
        print('\t|--- d=' + str(lennard_jones_d))
        print('\t|--- U=' + str(lj))
        U_init +=lj
    print('Total')
    print('\t|--- U=' + str(U_init))


def plot_structure(structures, names=None, save=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    legend=[]
    if isinstance(structures, list):
        for i in range(len(structures)):
            ax.plot(structures[i][:, 0], structures[i][:, 1], structures[i][:, 2])
            if names is not None:
                legend.append(names[i])
    else:
        ax.plot(structures[:, 0], structures[:, 1], structures[:, 2])
        if names is not None:
            legend.append(names)
    ax.legend(legend)
    if save is not None:
        fig.savefig(save)



def internal_to_cartesian(internal, first):
    n_atoms = internal.shape[0] + 3
    cartesian = np.zeros((n_atoms,3))
    cartesian[:3] = first
    for i in range(3,n_atoms):
        A = cartesian[i-3]
        B = cartesian[i-2]
        C = cartesian[i-1]
        AB = B-A
        BC = C-B
        bc = BC/np.linalg.norm(BC)
        n = np.cross(AB, bc) / np.linalg.norm(np.cross(AB, bc))

        bond = internal[i-3,0]
        angle = internal[i-3,1]
        torsion = internal[i-3,2]

        M1 = generate_rotation_matrix(angle, n)
        M2 = generate_rotation_matrix(torsion, bc)


        D0 = bond*bc
        D1 = np.dot(M1, D0)
        D2 = np.dot(M2, D1)

        cartesian[i] = D2 + C

    return cartesian

def cartesian_to_internal(x):

    bonds=np.linalg.norm(x[1:] - x[:-1], axis=1)

    angles = np.arccos(np.sum((x[:-2] - x[1:-1]) * (x[1:-1] - x[2:]), axis=1)
                      / (np.linalg.norm(x[:-2] - x[1:-1], axis=1) * np.linalg.norm(x[1:-1] - x[2:], axis=1)))

    u1 =x[1:-2]-x[:-3]
    u2 =x[2:-1]-x[1:-2]
    u3 =x[3:]-x[2:-1]
    torsions=np.arctan2(np.linalg.norm(u2, axis=1) * np.sum(u1* np.cross(u2, u3), axis=1),
                        np.sum(np.cross(u1,u2)* np.cross(u2,u3), axis=1))

    return np.array([bonds[2:], angles[1:] , torsions]).T

def generate_rotation_matrix(angle, vector):
    ux, uy, uz = vector
    c = np.cos(angle)
    s = np.sin(angle)
    M= np.array([[ ux*ux*(1-c) + c   , ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                 [ ux*uy*(1-c) + uz*s, uy*uy*(1-c) + c   , uy*uz*(1-c) - ux*s],
                 [ ux*uz*(1-c) - uy*s, uy*uz*(1-c) + ux*s, uz*uz*(1-c) + c   ]])
    return M


