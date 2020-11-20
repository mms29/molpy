import numpy as np
import pystan
import pickle
import os.path

def read_pdb(file, pattern='CA'):
    x = []
    ca=[]
    with open(file, "r") as file :
        for line in file:
            l = line.split()
            if len(l) >0:
                if l[0] == 'ATOM':
                    x.append([float(l[6]), float(l[7]), float(l[8])])
                    if l[2] == pattern:
                        ca.append(len(x))

    x = np.array(x)
    return x, ca

def read_modes(file, n_modes=20):
    A = []
    for i in range(n_modes):
        A.append(np.loadtxt(file + str(i + 1)))
    return np.transpose(np.array(A),(1,0,2))

def read_stan_model(model, save=True, build=False, threads=0):
    if not os.path.exists('stan/'+model+'.pkl'):
        build=True
    if build==True:
        s=""
        with open('stan/'+model+'.stan', "r") as f:
            for line in f:
                s+=line
        if threads >0:
            extra_compile_args = ['-pthread', '-DSTAN_THREADS']
            sm = pystan.StanModel(model_code=s, extra_compile_args=extra_compile_args)
        else :
            sm = pystan.StanModel(model_code=s)
        if save==True:
            with open('stan/'+model+'.pkl', 'wb') as f:
                pickle.dump(sm, f)
    else:
        sm = pickle.load(open('stan/'+model+'.pkl', 'rb'))
    return sm

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
    return (1/((2*np.pi*(sigma**2))**(3/2)))*np.exp(-((1/(2*(sigma**2))) * (np.linalg.norm(x-mu)**2)))

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


def md_energy_minimization(x, sigma_md, U_lim, k_r, r_md, k_theta, theta_md, k_lj, d_lj):

    U_init = md_bonds_potential(x, k_r, r_md) \
             + md_angles_potential(x, k_theta, theta_md) \
             # + md_lennard_jones_potential(x, k_lj, d_lj)
    print("U init : " + str(U_init))

    s_md = 0
    n_atoms = x.shape[0]
    x_init = np.array(x)

    while (U_init > U_lim):

        # new direction
        x_md = np.random.normal(0, sigma_md, x_init.shape)
        x_tmp = x_md + x_init

        U_bonds = md_bonds_potential(x_tmp, k_r, r_md)
        U_angles = md_angles_potential(x_tmp, k_theta, theta_md)
        U_lennard_jones = md_lennard_jones_potential(x_tmp, k_lj, d_lj)
        U_md= U_bonds+U_angles+U_lennard_jones
        if (U_init > U_md):
            U_init = U_md
            x_init = x_tmp
            print("yes : " + str(U_md)+" = "+str(U_bonds) + " + "+str(U_angles)+ " + "+str(U_lennard_jones))
            s_md += np.var(x_md)
    return x_init,s_md

def md_bonds_potential(x, k_r, r0):
    U=0.0
    n_atoms = x.shape[0]
    for i in range(n_atoms - 1):
        r = np.linalg.norm(x[i] - x[i + 1])
        U += k_r * (r - r0) ** 2
    return U;

def md_angles_potential(x, k_theta, theta0):
    U = 0.0
    n_atoms = x.shape[0]

    for i in range(n_atoms - 2):
        theta = np.arccos(np.dot(x[i] - x[i + 1], x[i + 1] - x[i + 2])
                          / (np.linalg.norm(x[i] - x[i + 1]) * np.linalg.norm(x[i + 1] - x[i + 2])))
        U += k_theta * (theta - theta0) ** 2
    return U

def md_lennard_jones_potential(x, k_lj, d_lj):
    U = 0.0
    n_atoms = x.shape[0]
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                U += 4 * k_lj * (
                            (d_lj / np.linalg.norm(x[i] - x[j])) ** 12 - (d_lj / np.linalg.norm(x[i] - x[j])) ** 6)
    return U