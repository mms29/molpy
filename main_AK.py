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
n_iter = 1000
n_step = 50
n_warmup = 900
params1 ={
    "initial_biasing_factor" : 100,"local_dt" : 2e-15,
    "n_step": n_step, "n_iter":n_iter, "n_warmup":n_warmup,
    "potentials" : ["bonds", "angles", "dihedrals"],
    "target_coords":target.coords,
}
fit1  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params1, n_chain=2, verbose=0, prefix="results/AK_test1")
params2 ={
    "initial_biasing_factor" : 10,"local_dt" : 2e-15,
    "n_step": n_step, "n_iter":n_iter, "n_warmup":n_warmup,
    "potentials" : ["bonds", "angles", "dihedrals"],
    "target_coords":target.coords,
}
fit2  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params2, n_chain=2, verbose=0, prefix="results/AK_test2")

params3 ={
    "initial_biasing_factor" : 1,"local_dt" : 2e-15,
    "n_step": n_step, "n_iter":n_iter, "n_warmup":n_warmup,
    "potentials" : ["bonds", "angles", "dihedrals"],
    "target_coords":target.coords,
}
fit3  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params3, n_chain=2, verbose=0, prefix="results/AK_test3")
params4 ={
    "initial_biasing_factor" : 0.1,"local_dt" : 2e-15,
    "n_step": n_step, "n_iter":n_iter, "n_warmup":n_warmup,
    "potentials" : ["bonds", "angles", "dihedrals"],
    "target_coords":target.coords,
}
fit4  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params4, n_chain=2, verbose=0, prefix="results/AK_test4")

fits=  multiple_fitting(models=[fit1,fit2,fit3,fit4], n_chain=2, n_proc =10)
# fit.HMC()
# src.viewers.fit_potentials_viewer(fit)
# fit.show()
# fit.show_3D()
# # chimera_molecule_viewer([fit.res["mol"], init])

# data= []
# for j in [[i.flatten() for i in n["coord"]] for n in fit.fit]:
#     data += j
# src.functions.compute_pca(data=data, length=[len(data)], n_components=2)