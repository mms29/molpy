import mkl
mkl.set_num_threads(1)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from src.molecule import Molecule
from src.simulation import nma_deform
from src.flexible_fitting import *
from src.viewers import molecule_viewer, chimera_molecule_viewer, chimera_fit_viewer
from src.density import Image
from src.constants import *

########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
init =Molecule("data/AK/AK_PSF.pdb")
init.center()
fnModes = np.array(["data/AK/modes_psf/vec."+str(i+7) for i in range(3)])
init.set_normalModeVec(fnModes)
init.rotate([0,np.pi/2,0])

# init.set_forcefield(psf_file="data/AK/AK.psf", prm_file= "data/toppar/par_all36_prot.prm")
init.allatoms2carbonalpha()
init.set_forcefield()
init.get_energy(potentials=["bonds", "angles", "dihedrals"])

# target = Molecule("data/1AKE/1ake_chainA_psf.pdb")
# target.center()
# target.allatoms2carbonalpha()
# target=init.copy()
# target.rotate([0.1,0.2,0.1])
# target.coords += 2
target = nma_deform(init, [300,-100,0])


size=64
sampling_rate=1.5
cutoff= 6.0
gaussian_sigma=2
target_density = Image.from_coords(coord=target.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, cutoff=cutoff)
init_density = Image.from_coords(coord=init.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, cutoff=cutoff)

init_density.show()
target_density.show()
########################################################################################################
#               HMC
########################################################################################################

params ={
    "biasing_factor" : 1,
    "local_dt" : 2e-15,
    "global_dt": 0.2,
    "rotation_dt": 0.0002,
    "shift_dt": 0.002,
    "n_step": 20,
    "n_iter":50,
    "n_warmup":45,
    "potentials" : ["bonds", "angles", "dihedrals"],
    "target":target,
    "limit" : None,
    "nb_update":20,
    "criterion":False
}
n_chain=4
verbose =2
fit  =FlexibleFitting(init = init, target= target_density, vars=[FIT_VAR_GLOBAL], params=params, n_chain=n_chain, verbose=verbose)
fit.HMC_chain()

fit.show()

chimera_molecule_viewer([fit.res["mol"], target])
chimera_molecule_viewer([target, init])

# data= []
# for j in [[i.flatten() for i in n["coord"]] for n in fit.fit]:
#     data += j
# src.functions.compute_pca(data=data, length=[len(data)], n_components=2)