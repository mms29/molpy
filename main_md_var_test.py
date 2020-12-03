import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting

########################################################################################################
#               IMPORT FILES
########################################################################################################



# import PDB
atoms, ca = src.functions.read_pdb("data/AK/AK.pdb")
modes = src.functions.read_modes("data/AK/modes/vec.", n_modes=5)[ca]
atoms= src.functions.center_pdb(atoms)[ca]
sim = src.simulation.Simulator(atoms)

nma_structure = sim.run_nma(modes = modes, amplitude=[100,-50,150,-100,50])
# rotated_structure = sim.rotate_pdb()
md_structure=  sim.run_md(U_lim=0.01, step=0.01, bonds={"k":0.001}, angles={"k":0.01}, lennard_jones={"k":1e-8, "d":3})
# sim.plot_structure(nma_structure)

n_voxels=12
gaussian_sigma = 2
sampling_rate = 128/n_voxels
sim.compute_density(size=n_voxels, sigma=gaussian_sigma, sampling_rate=sampling_rate)

########################################################################################################
#               FLEXIBLE FITTING
########################################################################################################

# n_shards=2
# os.environ['STAN_NUM_THREADS'] = str(n_shards)
N =40
n_md_var = np.array([ round(0.1*(1.2**i),6) for i in range(N)])
print(n_md_var)

fit_nma = []
fit_md_nma = []
fit_md = []
for i in range(N):
    print("///////////////////////////////////////////////////////////")
    print("///////////////////////////////////////////////////////////")
    print("///////////////////////////////////////////////////////////")
    print("ITER = "+str(i))
    print("///////////////////////////////////////////////////////////")
    print("///////////////////////////////////////////////////////////")
    print("///////////////////////////////////////////////////////////")


    input_data = {
        # structure
                 'n_atoms': sim.n_atoms,
                 'n_modes': modes.shape[1],
                 'y': sim.deformed_structure,
                 'x0': sim.init_structure,
                 'A': modes,
                 'sigma':200,
                 'epsilon': np.max(sim.deformed_density)/10,
                 'mu':0,

        # Energy
                 'U_init':1,
                 's_md':n_md_var[i],
                 'k_r':sim.bonds_k,
                 'r0':sim.bonds_r0,
                 'k_theta':sim.angles_k,
                 'theta0':sim.angles_theta0,
                 'k_lj':sim.lennard_jones_k,
                 'd_lj':sim.lennard_jones_d,

        # EM density
                 'N':sim.n_voxels,
                 'halfN':int(sim.n_voxels/2),
                 'gaussian_sigma':gaussian_sigma,
                 'sampling_rate': sampling_rate,
                 'em_density': sim.deformed_density
                }


    fit_md_nma.append(src.fitting.Fitting(input_data, "md_nma_emmap"))
    fit_md_nma[i].optimizing(n_iter=10000)

    fit_md.append(src.fitting.Fitting(input_data, "md_emmap"))
    fit_md[i].optimizing(n_iter=10000)

    def comp_err(fit):
        opt_err=[]
        for i in fit:
            opt_structure =i.opt_results['x']
            opt_err.append(np.mean(np.linalg.norm(opt_structure -i.input_data['y'], axis=1)))
        return np.array(opt_err)

    fig, ax= plt.subplots(1,2, figsize=(20,8))

    opt_md_nma_time = [i.opt_time for i in fit_md_nma]
    opt_md_time =     [i.opt_time for i in fit_md]
    ax[0].plot(n_md_var[:i+1],opt_md_nma_time)
    ax[0].plot(n_md_var[:i+1],opt_md_time)
    ax[0].set_xlabel('Number of atoms')
    ax[0].set_ylabel('Time (s)')
    ax[0].legend(["md_nma", "md"])
    ax[0].set_title("Optimisation Time")


    opt_md_nma_err = comp_err(fit_md_nma)
    opt_md_err = comp_err(fit_md)

    ax[1].plot(n_md_var[:i+1], opt_md_nma_err)
    ax[1].plot(n_md_var[:i+1], opt_md_err)
    ax[1].set_xlabel('Number of atoms')
    ax[1].set_ylabel('RMSE')
    ax[1].legend(["md_nma", "md"])
    ax[1].set_title("Optimisation Error")
    fig.savefig("results/md_var_test.png")