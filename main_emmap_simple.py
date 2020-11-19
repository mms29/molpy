import pystan
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from functions import *


def to2Dmap(x, N):
    x_map = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            x_map[i,j]=np.exp(-(np.linalg.norm(x-np.array([i,j]))**2))
    return x_map

N=10
x = np.random.uniform(N/4,N - N/4, 2)
x_map = to2Dmap(x,N)




sm = read_stan_model("emmap_simple", build=False)


model_dat = {
    'N':N,
    'x_map':x_map,
    'sigma': 10,
    'epsilon': 5,
    'mu' : [N/2, N/2]
}
fit = sm.sampling(data=model_dat, iter=1000, warmup=800, chains=4)
la = fit.extract(permuted=True)
x_res = la['x']

n_iter = int(x_res.shape[0]/4)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(x_map)
for i in range(4):
    ax.plot(x_res[i*n_iter:(i+1)*n_iter,1], x_res[i*n_iter:(i+1)*n_iter,0], 'x')