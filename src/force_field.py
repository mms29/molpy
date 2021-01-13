import numpy as np
import src.constants

def get_energy(mol):

    print("ENERGY COMPUTATION")

    # bond energy
    U_bonds = get_bonds_energy(mol)
    print("|-- Bonds = "+str(round(U_bonds,2)))

    # angles energy
    U_angles = get_angles_energy(mol)
    print("|-- Angles = " + str(round(U_angles,2)))

    # torsions energy
    U_torsions = get_torsions_energy(mol)
    print("|-- Torsions = " + str(round(U_torsions,2)))

    # Van der Waals energy
    U_vdw = get_wdv_energy(mol)
    print("|-- Van der Waals = " + str(round(U_vdw,2)))

    # Van der Waals energy
    U_total = U_bonds + U_angles + U_torsions +U_vdw
    print("|== TOTAL = " + str(round(U_total,2)))

def get_bonds_energy(mol):
    return np.sum( src.constants.K_BONDS * np.square(mol.bonds - src.constants.R0_BONDS))

def get_angles_energy(mol):
    return np.sum(src.constants.K_ANGLES * np.square(mol.angles - src.constants.THETA0_ANGLES))

def get_torsions_energy(mol):
    return np.sum(src.constants.K_TORSIONS*(1 + np.cos(src.constants.N_TORSIONS*(mol.torsions) - src.constants.DELTA_TORSIONS)))

def get_wdv_energy(mol):
    dist = np.linalg.norm(np.repeat(mol.coords, mol.n_atoms, axis=0).reshape(mol.n_atoms, mol.n_atoms, 3) - mol.coords, axis=2)
    np.fill_diagonal(dist, np.nan)
    inv = src.constants.D_VDW/dist
    return np.nansum(4 * src.constants.K_VDW * (inv**12 - inv**6))


# CARTESIAN GRADIENT
def get_gradient(mol):
    print("GRADIENT COMPUTATION")
    F_bonds = get_bonds_gradient(mol)
    f = np.mean(np.linalg.norm(F_bonds,axis=1))
    print("|-- Bonds = " + str(round(f,2)) + "  -- dU = "+str(round(np.sum(F_bonds),2)))

    F_angles = get_angles_gradient(mol)
    f=np.mean(np.linalg.norm(F_angles,axis=1))
    print("|-- Angles = " + str(round(f,2))+ "  -- dU = "+str(round(np.sum(F_angles),2)))

    F_torsions = get_torsions_gradient(mol)
    f=np.mean(np.linalg.norm(F_torsions,axis=1))
    print("|-- Torsions = " + str(round(f,2))+ "  -- dU = "+str(round(np.sum(F_torsions),2)))

    F_total = F_bonds+ F_angles + F_torsions
    f=np.mean(np.linalg.norm(F_total,axis=1))
    print("|== TOTAL = " + str(round(f,2))+ "  -- dU = "+str(round(np.sum(F_total),2)))

    return F_total


def get_bonds_gradient(mol):
    f = np.zeros((mol.n_atoms,3))
    for i in range(mol.n_atoms-1):
        a= mol.coords[i]
        b= mol.coords[i+1]
        r=np.linalg.norm(b-a)
        fa = -2*src.constants.K_BONDS * (r - src.constants.R0_BONDS) * (r/np.linalg.norm(r))
        f[i] += fa
        f[i+1] += -fa
    return f

def get_angles_gradient(mol):
    f = np.zeros((mol.n_atoms,3))
    for i in range(mol.n_atoms-2):
        a= mol.coords[i]
        b= mol.coords[i+1]
        c= mol.coords[i+2]
        ab = b-a
        bc = c-b
        theta = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))
        h = -2*src.constants.K_ANGLES *(theta-src.constants.THETA0_ANGLES)
        pa = np.cross(-ab, np.cross(-ab, bc))
        pc = np.cross(-bc, np.cross(-ab, bc))
        fa = (h/np.linalg.norm(ab))*(pa/np.linalg.norm(pa))
        fc = (h/np.linalg.norm(bc))*(pc/np.linalg.norm(pc))
        f[i] += fa
        f[i+1] += -fa - fc
        f[i+2] += fc
    return f

def get_torsions_gradient(mol):
    f = np.zeros((mol.n_atoms,3))
    for i in range(mol.n_atoms-3):
        a= mol.coords[i]
        b= mol.coords[i+1]
        c= mol.coords[i+2]
        d= mol.coords[i+3]

        F = a-b
        G = b-c
        H = d-c
        A = np.cross(F,G)
        B = np.cross(H,G)

        cos =  np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
        sin =  np.dot(np.cross(B, A), G)/(np.linalg.norm(A)*np.linalg.norm(B)*np.linalg.norm(G))
        phi = -np.arctan(sin/cos)
        dU = - src.constants.K_TORSIONS*src.constants.N_TORSIONS*np.sin(src.constants.N_TORSIONS*phi - src.constants.DELTA_TORSIONS)

        A2 = np.dot(A,A)
        B2 = np.dot(B,B)
        AG = (np.linalg.norm(G)/(A2))*A
        AFG = (np.dot(F,G)/(np.linalg.norm(G)*A2))*A
        BG = (np.linalg.norm(G)/(B2))*B
        BHG = (np.dot(H,G)/(np.linalg.norm(G)*B2))*B

        fa = -AG
        fb = AG + AFG - BHG
        fc = BHG - AFG - BG
        fd = BG

        f[i]   += dU*fa
        f[i+1] += dU*fb
        f[i+2] += dU*fc
        f[i+3] += dU*fd

    return f




#INTERNAL GRADIENT
def get_gradient_i(mol):
    print("GRADIENT COMPUTATION")
    F_bonds = get_bonds_gradient_i(mol)
    F_angles = get_angles_gradient_i(mol)
    F_torsions = get_torsions_gradient_i(mol)
    return F_bonds, F_angles, F_torsions

def get_bonds_gradient_i(mol):
    return 2*src.constants.K_BONDS*(mol.bonds - src.constants.R0_BONDS)

def get_angles_gradient_i(mol):
    return 2*src.constants.K_ANGLES*(mol.angles - src.constants.THETA0_ANGLES)

def get_torsions_gradient_i(mol):
    return -src.constants.K_TORSIONS*src.constants.N_TORSIONS*(np.sin(src.constants.N_TORSIONS*(mol.torsions) - src.constants.DELTA_TORSIONS))