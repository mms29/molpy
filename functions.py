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

def read_stan_model(file, save=True, build=False):
    if build==True:
        s=""
        with open(file+'.stan', "r") as f:
            for line in f:
                s+=line
        sm = pystan.StanModel(model_code=s)
        if save==True:
            with open(file+'.pkl', 'wb') as f:
                pickle.dump(sm, f)
    else:
        sm = pickle.load(open(file+'.pkl', 'rb'))
    return sm
