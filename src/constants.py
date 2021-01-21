import numpy as np

K_BOLTZMANN = 1.380649e-23
CARBON_MASS = 1.99264671e-26

K_BONDS = 305.000
R0_BONDS= 3.796562314087224

K_ANGLES = 40.000
THETA0_ANGLES = np.pi*3/7 #np.pi/3

K_TORSIONS = 3.100
N_TORSIONS = 2
DELTA_TORSIONS = -np.pi/2

K_VDW = 1e-8
D_VDW = 3

DEFAULT_INPUT_DATA ={
    "k_bonds": K_BONDS,
    "r0_bonds": R0_BONDS,

    "theta0_angles": THETA0_ANGLES,
    "k_angles": K_ANGLES,

    "k_torsions": K_TORSIONS,
    "n_torsions": N_TORSIONS,
    "delta_torsions": DELTA_TORSIONS,

    "k_vdw" : K_VDW,
    "d_vdw" : D_VDW,
}

