import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting

########################################################################################################
#               IMPORT FILES
########################################################################################################



# import PDB
atoms, ca = src.functions.read_pdb("data/AK/AK.pdb")
modes = src.functions.read_modes("data/AK/modes/vec.", n_modes=5)[::2]
atoms= src.functions.center_pdb(atoms)[::2]
sim = src.simulation.Simulator(atoms)

nma_structure = sim.run_nma(modes = modes, amplitude=[100,-50,150,-100,50])
# rotated_structure = sim.rotate_pdb()
md_structure=  sim.run_md(U_lim=6, step=0.01, bonds={"k":0.001}, angles={"k":0.01}, lennard_jones={"k":1e-8, "d":3})
# sim.plot_structure(nma_structure)


# gaussian_sigma=2
# sampling_rate=8
# sim.compute_density(size=16, sigma=gaussian_sigma, sampling_rate=sampling_rate)
# sim.plot_density()

########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

# n_shards=2
# os.environ['STAN_NUM_THREADS'] = str(n_shards)
N = 3
n_atoms = (np.arange(N)+1)*10
fit_nma = []
fit_md_nma = []
fit_md = []
for i in range(N):
    input_data = {
        # structure
                 'n_atoms': n_atoms[i],
                 'n_modes': modes.shape[1],
                 'y': sim.deformed_structure[:n_atoms[i]],
                 'x0': sim.init_structure[:n_atoms[i]],
                 'A': modes[:n_atoms[i]],
                 'sigma':200,
                 'epsilon': 1, #np.max(sim.deformed_density)/10,
                 'mu':0,

        # Energy
                 'U_init':sim.U_lim,
                 's_md':20,
                 'k_r':sim.bonds_k,
                 'r0':sim.bonds_r0,
                 'k_theta':sim.angles_k,
                 'theta0':sim.angles_theta0,
                 'k_lj':sim.lennard_jones_k,
                 'd_lj':sim.lennard_jones_d,

        # EM density
        #          'N':sim.n_voxels,
        #          'halfN':int(sim.n_voxels/2),
        #          'gaussian_sigma':gaussian_sigma,
        #          'sampling_rate': sampling_rate,
        #          'em_density': sim.deformed_density
                }


    fit_nma.append(src.fitting.Fitting(input_data, "nma"))
    fit_nma[i].sampling(n_iter=10, n_warmup=50, n_chain=4)
    fit_nma[i].optimizing(n_iter=1000)

    fit_md_nma.append(src.fitting.Fitting(input_data, "md_nma"))
    fit_md_nma[i].sampling(n_iter=10, n_warmup=50, n_chain=4)
    fit_md_nma[i].optimizing(n_iter=1000)

    fit_md.append(src.fitting.Fitting(input_data, "md"))
    fit_md[i].sampling(n_iter=10, n_warmup=50, n_chain=4)
    fit_md[i].optimizing(n_iter=1000)
    # vb = fit.vb(n_iter=10000)
    # fit.plot_lp()
    # fit.plot_nma(sim.q, save="results/modes_amplitudes.png")
    # fit.plot_structure(save="results/3d_structures.png")
    # fit.plot_error_map(N=sim.n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate, save="results/error_map.png")

def comp_err(fit):
    opt_err=[]
    spl_err=[]
    for i in fit:
        opt_structure =i.opt_results['x']
        sampling_structure = np.mean(i.sampling_results['x'], axis=0)
        opt_err.append(np.mean(np.linalg.norm(opt_structure -i.input_data['y'], axis=1)))
        spl_err.append(np.mean(np.linalg.norm(opt_structure -i.input_data['y'], axis=1)))
    return np.array(opt_err), np.array(spl_err)

fig, ax= plt.subplots(2,2, figsize=(20,15))
spl_nma_time = [i.sampling_time for i in fit_nma]
spl_md_nma_time = [i.sampling_time for i in fit_md_nma]
spl_md_time = [i.sampling_time for i in fit_md]
ax[0,0].plot(n_atoms,spl_nma_time)
ax[0,0].plot(n_atoms,spl_md_nma_time)
ax[0,0].plot(n_atoms,spl_md_time)
ax[0,0].set_xlabel('Number of atoms')
ax[0,0].set_ylabel('Time (s)')
ax[0,0].legend(["nma", "md_nma", "md"])
ax[0,0].set_title("Sampling Time")

opt_nma_time =    [i.opt_time for i in fit_nma]
opt_md_nma_time = [i.opt_time for i in fit_md_nma]
opt_md_time =     [i.opt_time for i in fit_md]
ax[0,1].plot(n_atoms,opt_nma_time)
ax[0,1].plot(n_atoms,opt_md_nma_time)
ax[0,1].plot(n_atoms,opt_md_time)
ax[0,1].set_xlabel('Number of atoms')
ax[0,1].set_ylabel('Time (s)')
ax[0,1].legend(["nma", "md_nma", "md"])
ax[0,1].set_title("Optimisation Time")


opt_nma_err, spl_nma_err = comp_err(fit_nma)
opt_md_nma_err, spl_md_nma_err = comp_err(fit_md_nma)
opt_md_err, spl_md_err = comp_err(fit_md)

ax[1,0].plot(n_atoms,spl_nma_err)
ax[1,0].plot(n_atoms,spl_md_nma_err)
ax[1,0].plot(n_atoms,spl_md_err)
ax[1,0].set_xlabel('Number of atoms')
ax[1,0].set_ylabel('RMSE')
ax[1,0].legend(["nma", "md_nma", "md"])
ax[1,0].set_title("Sampling Error")

ax[1,1].plot(n_atoms, opt_nma_err)
ax[1,1].plot(n_atoms, opt_md_nma_err)
ax[1,1].plot(n_atoms, opt_md_err)
ax[1,1].set_xlabel('Number of atoms')
ax[1,1].set_ylabel('RMSE')
ax[1,1].legend(["nma", "md_nma", "md"])
ax[1,1].set_title("Optimisation Error")
fig.savefig("results/speed.png")