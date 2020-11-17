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

def volume_from_pdb(x, size=(50,50,50), sigma=1, sampling_rate=1, precision=0.001):

    n_atoms = x.shape[0]
    em_density = np.zeros(size)
    center_transform = np.array([size[0]/2,size[1]/2,size[2]/2]).astype(int)

    guassian_range = 0
    while(gaussian_pdf(np.array([guassian_range*sampling_rate,0,0]),np.array([0,0,0]), sigma) > precision):
        guassian_range+=1


    for a in range(n_atoms):
        pos = np.rint((x[a]/sampling_rate) + center_transform).astype(int)
        for i in range(pos[0] - guassian_range , pos[0] + guassian_range + 1):
            if (i>=0 and i <size[0]):
                for j in range(pos[1] - guassian_range , pos[1] + guassian_range + 1):
                    if (j >= 0 and j < size[1]):
                        for k in range(pos[2] - guassian_range, pos[2] +guassian_range + 1):
                            if (k >= 0 and k < size[2]):
                                em_density[i, j, k] += gaussian_pdf((np.array([i,j,k]) - center_transform)*sampling_rate,x[a], sigma)

    return em_density


def gaussian_pdf(x, mu, sigma):
    return (1/((2*np.pi*(sigma**2))**(3/2)))*np.exp(-((1/(2*(sigma**2))) * (np.linalg.norm(x-mu)**2)))
    # return np.exp(-(np.linalg.norm(x - mu) ** 2))

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

