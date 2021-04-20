import mkl
mkl.set_num_threads(1)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from src.molecule import Molecule
from src.simulation import nma_deform
from src.flexible_fitting import *
from src.viewers import molecule_viewer, chimera_molecule_viewer, chimera_fit_viewer
from src.density import Volume
from src.constants import *

########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
init =Molecule("data/AK/AK_PSF.pdb")
init.center()
fnModes = np.array(["data/AK/modes_psf/vec."+str(i+7) for i in range(3)])
init.set_normalModeVec(fnModes)

init.set_forcefield(psf_file="data/AK/AK.psf", prm_file= "data/toppar/par_all36_prot.prm")
# init.allatoms2carbonalpha()
# init.set_forcefield()

target = Molecule("data/1AKE/1ake_chainA_psf.pdb")
target.center()
target.save_pdb("data/1AKE/1ake_center.pdb")
# target.coords += np.array([1.5,-2.0,-0.5])
# target.rotate([0.17,-0.13,0.23])

size=100
sampling_rate=2.0
cutoff= 6.0
gaussian_sigma=2
target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, cutoff=cutoff)
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, cutoff=cutoff)
# target_density.show()
# target_density.save_mrc(file="tests/tests_data/input/AK/ak_all_carbonalpha.mrc")
# target.save_pdb(file="tests/tests_data/input/AK/ak_all_carbonalpha.pdb")

# chimera_fit_viewer(init,target_density)

########################################################################################################
#               HMC
########################################################################################################

params ={
    # "initial_biasing_factor" : 50,
    "biasing_factor" : 0.26,
    "local_dt" : 2e-15,
    "global_dt": 0.1,
    "rotation_dt": 0.0001,
    "shift_dt": 0.001,
    "n_step": 20,
    "n_iter":400,
    "n_warmup":390,
    "potentials" : ["bonds", "angles", "dihedrals", "impropers","vdw", "elec"],
    "target_coords":target.coords,
}
n_chain=2
verbose =1
fitxdt2  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fit_x_dt2")
fitadt2  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fit_a_dt2")
fitqdt2  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_GLOBAL,FIT_VAR_ROTATION,  FIT_VAR_SHIFT], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fit_q_dt2")

params["local_dt"] = 1e-15
fitxdt1  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params, n_chain=n_chain, verbose=verbose,
                      prefix="results/AK/fit_x_dt1")
fitadt1  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                      prefix="results/AK/fit_a_dt1")


params["potentials"] = ["bonds", "angles", "dihedrals"]
fitxnoNb1  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fit_x_noNb1")
fitanoNb1  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fit_a_noNb1")

params["local_dt"] = 2e-15
fitxnoNb2  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fit_x_noNb1")
fitanoNb2  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL, FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fit_a_noNb1")

params["local_dt"] = 2e-15
params["potentials"] = ["bonds", "angles", "dihedrals", "impropers","vdw", "elec"]
params["n_step"] = 10
params["n_iter"] = 800
params["n_warmup"] = 790

fitxnstep10  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fit_x_nstep10")
params["n_step"] = 40
params["n_iter"] = 200
params["n_warmup"] = 190

fitxnstep40  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fit_x_nstep40")

fits=  multiple_fitting(models=[fitxdt2 , fitadt2 , fitqdt2 , fitxdt1 , fitadt1 , fitxnoNb1 , fitanoNb1 , fitxnoNb2 , fitanoNb2, fitxnstep10, fitxnstep40],
                        n_chain=n_chain, n_proc =25)
# fit.HMC()
# src.viewers.fit_potentials_viewer(fit)
# fit.show()
# fit.show_3D()
# # chimera_molecule_viewer([fit.res["mol"], init])

# data= []
# for j in [[i.flatten() for i in n["coord"]] for n in fit.fit]:
#     data += j
# src.functions.compute_pca(data=data, length=[len(data)], n_components=2)