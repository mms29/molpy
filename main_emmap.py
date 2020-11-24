import matplotlib.pyplot as plt
from src.functions import *

# import PDB
x, ca = read_pdb("data/AK/AK.pdb")
x=center_pdb(x[ca][:50])
n_atoms, _ = x.shape

# Read Modes
n_modes = 20
A = read_modes("data/AK/modes/vec.", n_modes=n_modes)[ca][:50]

#Simulate EM map
n_modes_fitted = 5
q = np.zeros(n_modes)
q[7:(7+n_modes_fitted)]=np.random.uniform(-200,200,n_modes_fitted)
y=np.zeros(x.shape)
for i in range(n_atoms):
    y[i] = np.dot(q ,A[i]) + x[i]

N = 12
sampling_rate=4
gaussian_sigma=2
em_density = volume_from_pdb(y, N, sigma=gaussian_sigma, sampling_rate=sampling_rate, precision=0.0001)
em_density2 = volume_from_pdb(x,N, sigma=gaussian_sigma, sampling_rate=sampling_rate, precision=0.0001)
em_vector = to_vector(em_density)

fig, ax =plt.subplots(1,2)
ax[0].imshow(em_density[int(N/2)])
ax[1].imshow(em_density2[int(N/2)])

# READ STAN MODEL
n_shards=2
os.environ['STAN_NUM_THREADS'] = str(n_shards)
sm = read_stan_model("nma_emmap_map_rect", build=False, threads=1)

model_dat = {'n_atoms': n_atoms,
             'n_modes':n_modes_fitted,
             'N':N,
             'em_density':to_vector(em_density),
             'x0':x,
             'A': A[:,7:(7+n_modes_fitted),:],
             'sigma':200,
             'epsilon':np.max(em_density)/10,
             'mu':np.zeros(n_modes_fitted),
             'sampling_rate':sampling_rate,
             'gaussian_sigma' :gaussian_sigma,
             'halfN': int(N/2),
             'n_shards':n_shards}
fit = sm.sampling(data=model_dat, iter=300, warmup=200, chains=4)
print("---- STAN END")
la = fit.extract(permuted=True )
q_res = la['q']
lp = la['lp__']
for i in range(n_modes_fitted):
    print(" q value "+str(i+7)+" : "+str(np.mean(q_res[:,i])))
# print(" q value 7 : "+str(np.mean(q_res)))
x_res = np.mean(la['x'], axis=0)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0],x[:,1], x[:,2], c='r')
ax.scatter(y[:,0],y[:,1], y[:,2], c='b')
ax.scatter(x_res[:,0], x_res[:,1], x_res[:,2], marker="x", c='g', s=100)
fig.savefig("results/3d_structures.png")
# fig.show()
print("---- PLOT1 END")


fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(n_modes_fitted):
    parts = ax.violinplot(q_res[:,i],[i], )
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_edgecolor('grey')
    for partname in ('cbars', 'cmins', 'cmaxes', ):
        vp = parts[partname]
        vp.set_edgecolor('#1f77b4')
    ax.plot(i,q[i+7], 'x', color='r')
fig.savefig("results/modes_amplitudes.png")
# fig.show()

# print("---- PLOT2 END")
#
# em_density_res = np.mean(la['sim_density'], axis=0)
# em_density_res_sim = volume_from_pdb(x_res, N, sigma=gaussian_sigma, sampling_rate=sampling_rate, precision=0.0001)
# fig, ax =plt.subplots(1,4)
# ax[0].imshow(em_density[int(N/2)])
# ax[1].imshow(em_density2[int(N/2)])
# ax[2].imshow(em_density_res[int(N/2)])
# ax[3].imshow(em_density_res_sim[int(N/2)])
#
# fig = plt.figure(figsize=(15,5))
# l=0.001
# ax1 = fig.add_subplot(141, projection='3d')
# ax2 = fig.add_subplot(142, projection='3d')
# ax3 = fig.add_subplot(143, projection='3d')
# ax4 = fig.add_subplot(144, projection='3d')
# ax1.voxels(em_density>l)
# ax2.voxels(em_density2>l)
# ax1.voxels(em_density_res>l)
# ax1.set_title("target")
# ax2.set_title("init")
# ax3.set_title("sim")
# ax4.set_title("sim_supposed")
