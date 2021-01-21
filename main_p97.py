import matplotlib.pyplot as plt
from src.functions import *
import src.simulation
import src.fitting
from src.viewers import structures_viewer
from src.flexible_fitting import *
import src.molecule
import src.io

########################################################################################################
#               IMPORT FILES
########################################################################################################

# import PDB
init =src.io.read_pdb("data/P97/5ftn.pdb")
init.add_modes("data/ATPase/modes/vec.", n_modes=10)
init = init.select_atoms(pattern='CA')


target =src.io.read_pdb("data/ATPase/1su4_rotated3.pdb")
target = target.select_atoms(pattern='CA')
