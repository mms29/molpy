import hashlib
import multiprocessing.pool
import os.path
import pickle

import autograd.numpy as npg
import matplotlib.pyplot as plt
import numpy as np
import pystan

from src.flexible_fitting import FlexibleFitting


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

def volume_from_pdb_fast4(coord, size, sigma, sampling_rate=1, threshold=2):
    vox, n_vox = select_voxels(coord, size, sampling_rate, threshold)
    n_atoms= coord.shape[0]
    vol = np.zeros((size, size, size))
    x = np.repeat(coord, n_vox ** 3).reshape(n_atoms, 3, n_vox, n_vox, n_vox)
    i=0
    mu = (np.mgrid[vox[i,0]:vox[i,0]+n_vox,
                     vox[i,1]:vox[i,1]+n_vox,
                     vox[i,2]:vox[i,2]+n_vox] - size / 2) * sampling_rate
    for i in range(n_atoms):
        vol[vox[i,0]:vox[i,0]+n_vox,
            vox[i,1]:vox[i,1]+n_vox,
            vox[i,2]:vox[i,2]+n_vox]+=np.exp(-np.square(np.linalg.norm(x[i] - mu, axis=0))/(2*(sigma ** 2)))
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

def image_from_pdb(coord, size, sampling_rate, sigma):
    image = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            mu = ((np.array([i, j]) - np.ones(2) * (size / 2)) * sampling_rate)
            image[i, j] = np.sum(np.exp(-np.square(np.linalg.norm(coord[:, :2] - mu, axis=1)) / (2 * (sigma ** 2))))
    return image

def image_from_pdb_fast(coord, size, sampling_rate, sigma):
    mu = (np.mgrid[0:size, 0:size] - size / 2) * sampling_rate
    x = np.repeat(coord[:,:2], size ** 2).reshape(coord.shape[0], 2, size, size)
    return np.sum(np.exp(-np.square(np.linalg.norm(x - mu, axis=1)) / (2 * (sigma ** 2))), axis=0)

def image_from_pdb_fast2(coord, size, sampling_rate, sigma):
    image = np.zeros((size, size))
    mu = (np.mgrid[0:size, 0:size] - size / 2) * sampling_rate
    n_atoms= coord.shape[0]
    for i in range(n_atoms):
        x = np.repeat(coord[i, :2], size ** 2).reshape(2, size, size)
        image+=np.exp(-np.square(np.linalg.norm(x - mu, axis=0))/(2*(sigma ** 2)))
    return image

def image_from_pdb_fast3(coord, size, sampling_rate, sigma, threshold):
    vox, n_pix = select_voxels(coord, size, sampling_rate, threshold)
    pix = vox[:,:2]
    n_atoms = coord.shape[0]
    img = np.zeros((size, size))
    for i in range(n_atoms):
        mu = (np.mgrid[pix[i, 0]:pix[i, 0] + n_pix,
              pix[i, 1]:pix[i, 1] + n_pix] - size / 2) * sampling_rate
        x = np.repeat(coord[i,:2], n_pix ** 2).reshape(2, n_pix, n_pix)
        img[pix[i, 0]:pix[i, 0] + n_pix,
              pix[i, 1]:pix[i, 1] + n_pix] += np.exp(-np.square(np.linalg.norm(x - mu, axis=0)) / (2 * (sigma ** 2)))

    return img


def get_grad_RMSD_img(coord, psim, pexp, size, sampling_rate, sigma):

    n_atoms = coord.shape[0]
    pdiff = psim - pexp
    dx = np.zeros(coord.shape)
    for i in range(size):
        for j in range(size):
            mu = ((np.array([i, j]) - np.ones(2) * (size / 2)) * sampling_rate)
            dpsim =-(1/(sigma**2)) * (coord[:,:2]-mu) * np.repeat(np.exp(-np.square(np.linalg.norm(coord[:,:2]-mu, axis=1))/(2*(sigma ** 2))),2).reshape(n_atoms,2)
            dx[:,:2] += 2* pdiff[i,j]*dpsim
    return dx

def get_grad_RMSD2_img_rot(coord, psim, pexp, size, sampling_rate, sigma, angles, coord0):
    # coord= x0*R
    # coord0 = x0
    n_atoms = coord.shape[0]
    pdiff = psim - pexp

    dcoord = np.zeros(coord.shape)
    dangles = np.zeros(3)
    mu = (np.mgrid[0:size, 0:size] - size / 2) * sampling_rate
    for i in range(n_atoms):
        x = np.repeat(coord[i, :2], size ** 2).reshape(2, size, size)
        tmp = 2* pdiff*np.exp(-np.square(np.linalg.norm(x-mu, axis=0))/(2*(sigma ** 2)))
        da, db, dc, dx, dy, dz = gradient_rotation(angles=angles, coord=coord0[i], mu=mu)
        dpsimx =-(1/(sigma**2)) * np.array([dx,dy,dz]) * np.array([tmp,tmp, tmp])
        dcoord[i] = np.sum(dpsimx, axis=(1,2))

        dpsimangles = -(1/(sigma**2)) * np.array([da,db,dc]) * np.array([tmp,tmp,tmp])
        dangles += np.sum(dpsimangles, axis=(1,2))
    return dcoord, dangles

def get_grad_RMSD2_img(coord, psim, pexp, size, sampling_rate, sigma):
    n_atoms = coord.shape[0]
    pdiff = psim - pexp

    dx = np.zeros(coord.shape)
    mu = (np.mgrid[0:size, 0:size] - size / 2) * sampling_rate
    for i in range(n_atoms):
        x = np.repeat(coord[i, :2], size ** 2).reshape(2, size, size)
        tmp = 2* pdiff*np.exp(-np.square(np.linalg.norm(x-mu, axis=0))/(2*(sigma ** 2)))
        dpsim =-(1/(sigma**2)) * (x-mu) * np.array([tmp,tmp])
        dx[i,:2] = np.sum(dpsim, axis=(1,2))
    return dx

def get_grad_RMSD3_img(coord, psim, pexp, size, sampling_rate, sigma,threshold):
    vox, n_pix = select_voxels(coord, size, sampling_rate, threshold)
    pix = vox[:, :2]
    n_atoms = coord.shape[0]
    pdiff = psim - pexp

    dx = np.zeros(coord.shape)
    for i in range(n_atoms):
        x = np.repeat(coord[i, :2], n_pix ** 2).reshape(2, n_pix, n_pix)
        mu = (np.mgrid[pix[i, 0]:pix[i, 0] + n_pix,
              pix[i, 1]:pix[i, 1] + n_pix] - size / 2) * sampling_rate
        tmp = 2* pdiff[pix[i, 0]:pix[i, 0] + n_pix,
              pix[i, 1]:pix[i, 1] + n_pix]*np.exp(-np.square(np.linalg.norm(x-mu, axis=0))/(2*(sigma ** 2)))
        dpsim =-(1/(sigma**2)) * (x-mu) * np.array([tmp,tmp])
        dx[i,:2] = np.sum(dpsim, axis=(1,2))
    return dx

def get_grad_RMSD_NMA_img(coord, psim, pexp, size, sampling_rate, sigma, A_modes):
    # NB : coord = x0 + q* A

    n_atoms = coord.shape[0]
    n_modes = A_modes.shape[1]
    pdiff = psim - pexp

    dx = np.zeros(coord.shape)
    dq = np.zeros(n_modes)
    mu = (np.mgrid[0:size, 0:size] - size / 2) * sampling_rate
    for i in range(n_atoms):
        x = np.repeat(coord[i,:2], size ** 2).reshape(2, size, size)
        tmp = 2* pdiff*np.exp(-np.square(np.linalg.norm(x-mu, axis=0))/(2*(sigma ** 2)))

        dpsimx =-(1/(sigma**2)) * (x-mu) * np.array([tmp,tmp])
        dx[i,:2] = np.sum(dpsimx, axis=(1,2))

        dpsimq = -(1 / (sigma ** 2)) * (x - mu) * np.array([tmp, tmp])
        dq += np.dot(A_modes[i, :,:2] , np.sum(dpsimq, axis=(1, 2)))

    return dx, dq

def get_grad_RMSD3_NMA_img(coord, psim, pexp, size, sampling_rate, sigma, A_modes, threshold):
    vox, n_pix = select_voxels(coord, size, sampling_rate, threshold)
    pix = vox[:, :2]

    n_atoms = coord.shape[0]
    n_modes = A_modes.shape[1]
    pdiff = psim - pexp

    dx = np.zeros(coord.shape)
    dq = np.zeros(n_modes)
    for i in range(n_atoms):
        x = np.repeat(coord[i,:2], n_pix ** 2).reshape(2, n_pix, n_pix)
        mu = (np.mgrid[pix[i, 0]:pix[i, 0] + n_pix,
              pix[i, 1]:pix[i, 1] + n_pix] - size / 2) * sampling_rate
        tmp = 2* pdiff[pix[i, 0]:pix[i, 0] + n_pix,
              pix[i, 1]:pix[i, 1] + n_pix]*np.exp(-np.square(np.linalg.norm(x-mu, axis=0))/(2*(sigma ** 2)))

        dpsimx =-(1/(sigma**2)) * (x-mu) * np.array([tmp,tmp])
        dx[i,:2] = np.sum(dpsimx, axis=(1,2))

        dpsimq = -(1 / (sigma ** 2)) * (x - mu) * np.array([tmp, tmp])
        dq += np.dot(A_modes[i, :,:2] , np.sum(dpsimq, axis=(1, 2)))

    return dx, dq

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

def gradient_rotation(angles, coord, mu):
    a,b,c =angles
    cos=np.cos
    sin=np.sin
    x = coord[0]
    y = coord[1]
    z = coord[2]
    u = mu[0]
    v = mu[1]

    ##################################################################################
    # f = (x * (cos(a) * cos(b)) + y * (sin(a) * cos(b)) - z * sin(b) - u) ^ 2 + (
    #             x * (cos(a) * sin(b) * sin(c) - sin(a) * cos(c)) + y * (
    #                 sin(a) * sin(b) * sin(c) + cos(a) * cos(c)) + z * (cos(b) * sin(c)) - v) ^ 2
    ################################################################################

    da = 2 * cos(b) * (y * cos(a) - x * sin(a)) * (cos(b) * (y * sin(a) + x * cos(a)) - sin(b) * z - u) + 2 * (
            y * (sin(b) * sin(c) * cos(a) - cos(c) * sin(a)) - x * (sin(b) * sin(c) * sin(a) + cos(c) * cos(a))) * (
                 y * (sin(b) * sin(c) * sin(a) + cos(c) * cos(a)) - x * (
                 cos(c) * sin(a) - sin(b) * sin(c) * cos(a)) + cos(b) * sin(c) * z - v)

    db = 2 * (-sin(c) * z * sin(b) + sin(a) * sin(c) * y * cos(b) + cos(a) * sin(c) * x * cos(b)) * (
            y * (sin(a) * sin(c) * sin(b) + cos(a) * cos(c)) + x * (
            cos(a) * sin(c) * sin(b) - sin(a) * cos(c)) + sin(c) * z * cos(b) - v) + 2 * (
                 -sin(a) * y * sin(b) - cos(a) * x * sin(b) - z * cos(b)) * (
                 -z * sin(b) + sin(a) * y * cos(b) + cos(a) * x * cos(b) - u)

    dc = -2 * ((cos(a) * y - sin(a) * x) * sin(c) + (-cos(b) * z - sin(a) * sin(b) * y - cos(a) * sin(b) * x) * cos(
        c)) * ((cos(b) * z + sin(a) * sin(b) * y + cos(a) * sin(b) * x) * sin(c) + (cos(a) * y - sin(a) * x) * cos(
             c) - v)

    dx = 2 * (cos(a) * sin(b) * sin(c) - sin(a) * cos(c)) * (
                (cos(a) * sin(b) * sin(c) - sin(a) * cos(c)) * x + cos(b) * sin(c) * z + (
                    sin(a) * sin(b) * sin(c) + cos(a) * cos(c)) * y - v) + 2 * cos(a) * cos(b) * (
                cos(a) * cos(b) * x - sin(b) * z + sin(a) * cos(b) * y - u)

    dy = 2 * (sin(a) * sin(b) * sin(c) + cos(a) * cos(c)) * (
                (sin(a) * sin(b) * sin(c) + cos(a) * cos(c)) * y + cos(b) * sin(c) * z + (
                    cos(a) * sin(b) * sin(c) - sin(a) * cos(c)) * x - v) + 2 * sin(a) * cos(b) * (
                sin(a) * cos(b) * y - sin(b) * z + cos(a) * cos(b) * x - u)

    dz = 2 * cos(b) * sin(c) * (cos(b) * sin(c) * z + (sin(a) * sin(b) * sin(c) + cos(a) * cos(c)) * y + (
                cos(a) * sin(b) * sin(c) - sin(a) * cos(c)) * x - v) - 2 * sin(b) * (
                -sin(b) * z + sin(a) * cos(b) * y + cos(a) * cos(b) * x - u)

    return da, db, dc, dx, dy, dz

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

        dpsim =-(1/(sigma**2)) * (x-mu) * np.array([tmp,tmp,tmp])
        dx[i] = np.sum(dpsim, axis=(1,2,3))
        dq += np.dot(A_modes[i] ,dx[i])

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

def generate_euler_matrix(angles):
    a, b, c = angles
    cos = npg.cos
    sin = npg.sin
    R = npg.array([[ cos(c) *  cos(b) * cos(a) -  sin(c) * sin(a), cos(c) * cos(b) * sin(a) +  sin(c) * cos(a), -cos(c) * sin(b)],
                  [- sin(c) *  cos(b) * cos(a) - cos(c) * sin(a), - sin(c) * cos(b) * sin(a) + cos(c) * cos(a), sin(c) * sin(b)],
                  [sin(b) * cos(a), sin(b) * sin(a), cos(b)]])
    return R
#
# def get_euler_autograd(coord,A , x, q, angles):
#     def rotate(coord, A, x, q, angles):
#         tmp = coord + x + npg.dot(q, A)
#         return npg.dot(generate_euler_matrix(angles), tmp.T).T
#     grad = elementwise_grad(rotate, (2,3,4))
#     return grad(coord,A,x, q, angles)

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

# def generate_euler_matrix2(angles):
#     a, b, c = angles
#     cos = np.cos
#     sin = np.sin
#     ca = cos(a)
#     sa = sin(a)
#     cb = cos(b)
#     sb = sin(b)
#     cg = cos(c)
#     sg = sin(c)
#     cc = cb * ca
#     cs = cb * sa
#     sc = sb * ca
#     ss = sb * sa
#     R = np.zeros((3, 3))
#     R[0, 0] = cg * cc - sg * sa
#     R[0, 1] = cg * cs + sg * ca
#     R[0, 2] = -cg * sb
#     R[1, 0] = -sg * cc - cg * sa
#     R[1, 1] = -sg * cs + cg * ca
#     R[1, 2] = sg * sb
#     R[2, 0] = sc
#     R[2, 1] = ss
#     R[2, 2] = cb
#     return R

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)

def multiple_fitting(init, targets, vars, n_chain, n_iter, n_warmup, params, n_proc):
    ff = []
    N = len(targets)
    for t in targets :
        ff.append(FlexibleFitting(init=init, target = t, vars=vars, n_chain=n_chain,
                                  n_iter=n_iter, n_warmup=n_warmup, params=params))
    ff = np.array(ff)

    n_loop = (N * n_chain) //n_proc
    n_last_process = ((N * n_chain) % n_proc)//n_chain
    n_process = n_proc//n_chain
    process = [np.arange(i*n_process, (i+1)*n_process) for i in range(n_loop)]
    process.append(np.arange(n_loop*n_process, n_loop*n_process + n_last_process))
    fits=[]
    print("Number of loops : "+str(n_loop))
    for i in process:

        print("\t fitting models # "+str(i))
        try :
            with NestablePool(n_process) as p:
                fits += p.map(FlexibleFitting.HMC, ff[i])
                p.close()
                p.join()
        except RuntimeError:
            print("Failed")
    return fits

