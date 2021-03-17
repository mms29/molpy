import matplotlib.pyplot as plt
import numpy as np

from src.molecule import Molecule
from src.simulation import nma_deform
import src.io
from src.viewers import chimera_fit_viewer
from src.density import Volume
from src.flexible_fitting import FlexibleFitting, multiple_fitting
from src.functions import compute_pca
from src.constants import *


########################################################################
#             INITIAL STRUCTURE
########################################################################

init =Molecule.from_file("data/AK_tomos/AK.pdb")
init.center_structure()
fnModes = np.array(["data/AK_tomos/modes/vec."+str(i+7) for i in range(3)])
init.add_modes(fnModes)
# init.set_forcefield(psf_file="data/AK/AK.psf")
init.select_atoms(pattern='CA')
init.set_forcefield()

########################################################################
#             DENSITY
########################################################################
N=1000
size=64
voxel_size=2.2
threshold= 4
gaussian_sigma=2
init_density = Volume.from_coords(coord=init.coords, size=size, voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
# test = "/home/guest/ScipionUserData/projects/synthesize/Runs/001946_FlexProtVolumeDenoise/extra/filtered/tmp/"
# gt_modes, gt_shifts, gt_angles = src.io.read_xmd("/home/guest/ScipionUserData/projects/synthesize/Runs/001661_FlexProtSynthesizeSubtomo/extra/GroundTruth.xmd")
test = "/scratch/cnt0028/imp0998/rvuillemot/AK_1000_denoised/ak_denoised/"
gt_modes, gt_shifts, gt_angles = src.io.read_xmd("/scratch/cnt0028/imp0998/rvuillemot/AK_1000_denoised/GroundTruth.xmd")


targets=[]
mols =[]
for i in range(N):
    t = Volume.from_file(file=test+str(i+1).zfill(5)+"_reconstructed.mrc",
                                    voxel_size=voxel_size, sigma=gaussian_sigma, threshold=threshold)
    t.rescale(init_density, method="normal")
    targets.append(t)
    mols.append(nma_deform(init, gt_modes[i]))

# targets[0].show()
# targets[0].compare_hist(init_density)
# chimera_fit_viewer(init, targets[0])

########################################################################################################
#               HMC
########################################################################################################

params ={
    "biasing_factor" : 100,
    "potential_factor" : 1,

    "local_dt" : 0.01,
    "global_dt" : 0.1,

    "n_iter":20,
    "n_warmup":10,
    "n_step": 20,
    "criterion": False,
}

n_chain = 4
n_proc = 80
models= []
for i in targets:
    models.append(FlexibleFitting(init=init, target=i, vars=["global", "local"], params=params,n_chain=n_chain, verbose=0))

# n=7
# models[n].HMC_chain()
# models[n].show()
# models[n].show_3D()
# print(models[n].res["global"])
# print(gt_modes[n])

fits = multiple_fitting(models, n_chain=n_chain, n_proc=n_proc,save_dir="/scratch/cnt0028/imp0998/rvuillemot/AK_1000_denoised/results/")

pca_data = [i.coords.flatten() for i in mols]+ [i.res["mol"].coords.flatten() for i in fits]
pca_length = [len(targets)]+ [ len(fits)]
pca_labels = ["Ground Truth"]+[ "Fitted"]
compute_pca(data=pca_data, length=pca_length, labels=pca_labels, n_components=2, save="/scratch/cnt0028/imp0998/rvuillemot/AK_1000_denoised/results/AK_denoised_pca.png")
compute_pca(data=pca_data, length=pca_length, labels=pca_labels, n_components=3, save="/scratch/cnt0028/imp0998/rvuillemot/AK_1000_denoised/results/AK_denoised_pca_3D.png")

