import mkl
mkl.set_num_threads(1)
from src.molecule import Molecule
from src.simulation import nma_deform
from src.flexible_fitting import *
from src.viewers import molecule_viewer, chimera_molecule_viewer
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

# init.set_forcefield(psf_file="data/AK/AK.psf", prm_file= "data/toppar/par_all36_prot.prm")
init.allatoms2carbonalpha()
init.set_forcefield()

q = [300,-100,0,]
target = nma_deform(init, q)
# target.coords += np.array([1.5,-2.0,-0.5])
# target.rotate([0.17,-0.13,0.23])

# molecule_viewer([target, init])

size=64
sampling_rate=1.5
threshold= 4
gaussian_sigma=2
target_density = Volume.from_coords(coord=target.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold)
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold)
# target_density.show()
# target_density.save_mrc(file="tests/tests_data/input/AK/ak_all_carbonalpha.mrc")
# target.save_pdb(file="tests/tests_data/input/AK/ak_all_carbonalpha.pdb")

# chimera_molecule_viewer([target, init])

########################################################################################################
#               HMC
########################################################################################################
params ={
    "initial_biasing_factor" : 50,
    "n_step": 10,

    "local_dt" : 2*1e-15,
    "temperature" : 1000,

    "global_dt" : 0.05,
    "rotation_dt" : 0.0001,
    "shift_dt" : 0.001,
    "n_iter":50,
    "n_warmup":25,
    "potentials" : ["bonds", "angles", "dihedrals"], #"vdw", "elec"],
    "cutoffnb": 20,
    "cutoffpl" :25,
    "target_coords":target.coords,
}
fit  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_LOCAL], params=params, n_chain=4, verbose=2)
fit.HMC()
fit.show()
fit.show_3D()
chimera_molecule_viewer([fit.res["mol"], target])

# data= []
# for j in [[i.flatten() for i in n["coord"]] for n in fit.fit]:
#     data += j
# src.functions.compute_pca(data=data, length=[len(data)], n_components=2)