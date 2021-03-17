K_BOLTZMANN = 1.380649e-23
CARBON_MASS = 1.99264671e-26

K_BONDS = 305.000
R0_BONDS= 3.796562314087224

K_ANGLES = 40.000
THETA0_ANGLES = 102.85 #np.pi*3/7 #np.pi/3

K_TORSIONS = 3.100
N_TORSIONS = 2
DELTA_TORSIONS = -90.0 # -np.pi/2

K_VDW = 1e-8
D_VDW = 3

ATOMIC_MASS_UNIT = 1.66054e-27

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

FIT_VAR_LOCAL = "local"
FIT_VAR_GLOBAL = "global"
FIT_VAR_ROTATION = "rotation"
FIT_VAR_SHIFT = "shift"

DEFAULT_FIT_PARAMS ={
    "biasing_factor" : 1,
    "potential_factor" : 1,

    "n_step": 10,
    "n_iter": 20,
    "n_warmup": 10,
    "criterion" :False,

    FIT_VAR_LOCAL+"_dt" : 0.001,
    FIT_VAR_LOCAL+"_mass" : 1,
    FIT_VAR_GLOBAL+"_dt" : 0.01,
    FIT_VAR_GLOBAL+"_mass" : 1,
    FIT_VAR_ROTATION+"_dt": 0.00001,
    FIT_VAR_ROTATION+"_mass": 1,
    FIT_VAR_SHIFT+"_dt": 0.00001,
    FIT_VAR_SHIFT+"_mass": 1,
}

TOPOLOGY_FILE = "/home/guest/toppar/top_all36_prot.rtf"
PARAMETER_FILE = "/home/guest/toppar/par_all36_prot.prm"

