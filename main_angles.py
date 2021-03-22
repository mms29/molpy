from src.molecule import Molecule
from src.simulation import nma_deform
from src.flexible_fitting import *
from src.viewers import molecule_viewer, chimera_molecule_viewer
from src.density import Volume
from src.constants import *
from src.functions import show_rmsd_fit

########################################################################################################
#               IMPORT FILES
########################################################################################################

N=1

# import PDB
init =Molecule.from_file("data/AK/AK.pdb")
init.center_structure()
# fnModes = np.array(["data/AK/modes/vec."+str(i+7) for i in range(3)])
# init.add_modes(fnModes)

# init.set_forcefield(psf_file="data/AK/AK.psf")
init.select_atoms(pattern='CA')
init.set_forcefield()

# q = [100,-100,0,]
# target = nma_deform(init, q)
# target.rotate([0.17,-0.13,0.23])

size=64
sampling_rate=2.2
threshold= 4
gaussian_sigma=2
angles = np.array([[0.5,0.2,-0.2]])
# angles = np.random.uniform(-np.pi/2, np.pi/2, (N,3))
targets=[]
target_densities=[]
for i in range(N):
    target = copy.deepcopy(init)
    target.rotate(angles[i])
    targets.append(target)
    target_densities.append(Volume.from_coords(coord=target.coords, size=size, voxel_size=sampling_rate, sigma=gaussian_sigma, threshold=threshold))

params ={
    "biasing_factor" : 100,
    "n_step": 20,
    "criterion": False,
    "n_iter":20,
    "n_warmup":10,
    "rotation_dt" : 0.0001,
    "rotation_mass": 10,}
n_chain = 4
verbose=2

models= []
for i in target_densities:
    models.append(FlexibleFitting(init=init, target=i, vars=["rotation"], params=params, n_chain=n_chain, verbose=verbose))

fit = models[0].HMC()

fit.show()

show_rmsd_fit(mol=targets[0], fit=fit)
