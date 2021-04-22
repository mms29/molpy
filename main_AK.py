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
    "initial_biasing_factor" : 50,
    "local_dt" : 2e-15,
    "global_dt": 0.1,
    "rotation_dt": 0.0001,
    "shift_dt": 0.001,
    "n_step": 5000,
    "n_iter":1,
    "n_warmup":0,
    "potentials" : ["bonds", "angles", "dihedrals", "impropers","vdw", "elec"],
    "target_coords":target.coords,
}
n_chain=4
verbose =2
fit1  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fitxbf50md")
params["initial_biasing_factor"]=10
fit2  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fitxbf10md")
params["local_dt"]=1e-15
fit3  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fitxbf10dt1md")
params["initial_biasing_factor"]=50
fit4  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params, n_chain=n_chain, verbose=verbose,
                       prefix="results/AK/fitxbf50dt1md")

fits=  multiple_fitting(models=[fit1, fit2, fit3, fit4],
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