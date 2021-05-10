import mkl
mkl.set_num_threads(1)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from src.molecule import Molecule
from src.flexible_fitting import *
from src.viewers import molecule_viewer, chimera_molecule_viewer, chimera_fit_viewer, ramachandran_viewer
from src.density import Volume
from src.constants import *

########################################################################################################
#               IMPORT FILES
########################################################################################################
# import PDB
init =Molecule("data/AK/AK_PSF.pdb")
init.center()
fnModes = np.array(["data/AK/modes_psf/vec."+str(i+10) for i in range(3)])
init.set_normalModeVec(fnModes)

init.set_forcefield(psf_file="data/AK/AK.psf", prm_file= "data/toppar/par_all36_prot.prm")
init.get_energy(verbose=True)
# init.allatoms2carbonalpha()
# init.set_forcefield()

target = Molecule("data/1AKE/1ake_chainA_psf.pdb")
target.center()
target.save_pdb("data/1AKE/1ake_center.pdb")


size=64
sampling_rate=2.0
cutoff= 6.0
gaussian_sigma=2
target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, cutoff=cutoff)
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, cutoff=cutoff)


########################################################################################################
#               HMC
########################################################################################################

params ={
    "biasing_factor" : 10000,
    "local_dt" : 2e-15,
    "global_dt": 0.1,
    "rotation_dt": 0.0001,
    "shift_dt": 0.001,
    "n_step": 10000,
    "n_iter":1,
    "n_warmup":0,
    "potentials" : ["bonds", "angles", "dihedrals", "impropers","urey", "elec", "vdw"],
    "target":target,
    "limit" : 100,
    "nb_update":20,
    "criterion":False,
    "gradient": "CC"
}
n_chain=1
verbose =2
fits=[]
params["biasing_factor"] = 1000
fits.append(FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fita_cc0_modes"))
params["biasing_factor"] = 5000
fits.append(FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fita_cc1_modes"))
params["biasing_factor"] = 10000
fits.append(FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fita_cc3_modes"))
params["biasing_factor"] = 50000
fits.append(FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fita_cc4_modes"))
params["biasing_factor"] = 100000
fits.append(FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fita_cc5_modes"))
params["gradient"] = "LS"
params["biasing_factor"] = 0.01
fits.append(FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fita_ls0_modes"))
params["biasing_factor"] = 0.03
fits.append(FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fita_ls1_modes"))
params["biasing_factor"] = 0.6
fits.append(FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fita_ls3_modes"))
params["biasing_factor"] = 0.1
fits.append(FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fita_ls4_modes"))
params["biasing_factor"] = 0.2
fits.append(FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fita_ls5"))
fits=  multiple_fitting(models=fits, n_chain=n_chain, n_proc =25)

# fit1.HMC_chain()
# src.viewers.fit_potentials_viewer(fit1)
# fit.show()
# fit.show_3D()
# # chimera_molecule_viewer([fit.res["mol"], init])

# data= []
# for j in [[i.flatten() for i in n["coord"]] for n in fit.fit]:
#     data += j
# src.functions.compute_pca(data=data, length=[len(data)], n_components=2)