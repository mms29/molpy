import numpy as np
import src.constants
import autograd.numpy as npg
from autograd import grad, jacobian, elementwise_grad


################################################################################################
#    Cartesian Potential Energy
################################################################################################
def get_energy(mol, verbose=True):
    verbose: print("ENERGY COMPUTATION")

    U_bonds = 0
    U_angles= 0
    U_torsions = 0
    for i in range(mol.n_chain):

        # bond energy
        U_bonds += get_bonds_energy(mol.get_chain(i))

        # angles energy
        U_angles += get_angles_energy(mol.get_chain(i))

        # torsions energy
        U_torsions += get_torsions_energy(mol.get_chain(i))

    # Van der waals
    # U_vdw = get_wdv_energy(mol)

    U_total = U_bonds + U_angles + U_torsions #+ U_vdw
    if verbose:
        print("|-- Bonds = " + str(round(U_bonds, 2)))
        print("|-- Angles = " + str(round(U_angles, 2)))
        print("|-- Torsions = " + str(round(U_torsions, 2)))
        # print("|-- Van der Waals = " + str(round(U_torsions, 2)))
        print("|== TOTAL = " + str(round(U_total, 2)))

    return U_total

def get_bonds_energy(coord):
    r = npg.linalg.norm(coord[1:] - coord[:-1], axis=1)
    return npg.sum(src.constants.K_BONDS * npg.square(r - src.constants.R0_BONDS))

def get_angles_energy(coord):
    theta = npg.arccos(npg.sum((coord[:-2] - coord[1:-1]) * (coord[1:-1] - coord[2:]), axis=1)
                       / (npg.linalg.norm(coord[:-2] - coord[1:-1], axis=1) * npg.linalg.norm(
        coord[1:-1] - coord[2:], axis=1)))
    return npg.sum(src.constants.K_ANGLES * npg.square(theta - src.constants.THETA0_ANGLES))

def get_torsions_energy(coord):
    u1 = coord[1:-2] - coord[:-3]
    u2 = coord[2:-1] - coord[1:-2]
    u3 = coord[3:] - coord[2:-1]
    torsions = npg.arctan2(npg.linalg.norm(u2, axis=1) * npg.sum(u1 * npg.cross(u2, u3), axis=1),
                       npg.sum(npg.cross(u1, u2) * npg.cross(u2, u3), axis=1))
    return npg.sum(src.constants.K_TORSIONS*(1 + npg.cos(src.constants.N_TORSIONS*(torsions) - src.constants.DELTA_TORSIONS)))

def get_wdv_energy(mol, cutoff=1.22):
    dist = np.linalg.norm(np.repeat(mol.coords, mol.n_atoms, axis=0).reshape(mol.n_atoms, mol.n_atoms, 3) - mol.coords, axis=2)
    dist = np.fill_diagonal(dist,100)
    dist = dist[dist<cutoff]
    inv = src.constants.D_VDW / dist
    return np.sum(4 * src.constants.K_VDW * (inv ** 12 - inv ** 6))

################################################################################################
#    Internal Potential Energy (BUG ?)
################################################################################################

#
# def get_energy_i(mol, verbose=True):
#
#     if verbose : print("ENERGY COMPUTATION")
#
#     # bond energy
#     U_bonds = get_bonds_energy_i(mol)
#     if verbose : print("|-- Bonds = "+str(round(U_bonds,2)))
#
#     # angles energy
#     U_angles = get_angles_energy_i(mol)
#     if verbose : print("|-- Angles = " + str(round(U_angles,2)))
#
#     # torsions energy
#     U_torsions = get_torsions_energy_i(mol)
#     if verbose : print("|-- Torsions = " + str(round(U_torsions,2)))
#
#     # Van der Waals energy
#     U_vdw = get_wdv_energy(mol)
#     if verbose : print("|-- Van der Waals = " + str(round(U_vdw,2)))
#
#     # Van der Waals energy
#     U_total = U_bonds + U_angles + U_torsions +U_vdw
#     if verbose : print("|== TOTAL = " + str(round(U_total,2)))
#
#     return U_total
#
# def get_bonds_energy_i(mol):
#     return np.sum( src.constants.K_BONDS * np.square(mol.bonds - src.constants.R0_BONDS))
#
# def get_angles_energy_i(mol):
#     return np.sum(src.constants.K_ANGLES * np.square(mol.angles - src.constants.THETA0_ANGLES))
#
# def get_torsions_energy_i(mol):
#     return np.sum(src.constants.K_TORSIONS*(1 + np.cos(src.constants.N_TORSIONS*(mol.torsions) - src.constants.DELTA_TORSIONS)))
#
# def get_wdv_energy(mol):
#     dist = np.linalg.norm(np.repeat(mol.coords, mol.n_atoms, axis=0).reshape(mol.n_atoms, mol.n_atoms, 3) - mol.coords, axis=2)
#     np.fill_diagonal(dist, np.nan)
#     inv = src.constants.D_VDW/dist
#     return np.nansum(4 * src.constants.K_VDW * (inv**12 - inv**6))

################################################################################################
#    Cartesian Gradient
################################################################################################
#
# def get_gradient(mol, verbose=True):
#     if verbose : print("GRADIENT COMPUTATION")
#
#     F_bonds = get_bonds_gradient(mol)
#     f = np.mean(np.linalg.norm(F_bonds,axis=1))
#     if verbose : print("|-- Bonds = " + str(round(f,2)) + "  -- dU = "+str(round(np.sum(F_bonds),2)))
#
#     F_angles = get_angles_gradient(mol)
#     f=np.mean(np.linalg.norm(F_angles,axis=1))
#     if verbose : print("|-- Angles = " + str(round(f,2))+ "  -- dU = "+str(round(np.sum(F_angles),2)))
#
#     F_torsions = get_torsions_gradient(mol)
#     f=np.mean(np.linalg.norm(F_torsions,axis=1))
#     if verbose : print("|-- Torsions = " + str(round(f,2))+ "  -- dU = "+str(round(np.sum(F_torsions),2)))
#
#     F_total = F_bonds+ F_angles + F_torsions
#     f=np.mean(np.linalg.norm(F_total,axis=1))
#     if verbose : print("|== TOTAL = " + str(round(f,2))+ "  -- dU = "+str(round(np.sum(F_total),2)))
#
#     return F_total
#
#
# def get_bonds_gradient(mol):
#     n_atoms = mol.n_atoms
#     f = np.zeros((n_atoms,3))
#     dx = mol.coords[1:] - mol.coords[:-1]
#     r = np.linalg.norm(dx, axis=1)
#
#     fa = -2*src.constants.K_BONDS * dx * np.array([(1 - src.constants.R0_BONDS / r),
#                                                    (1 - src.constants.R0_BONDS / r),
#                                                    (1 - src.constants.R0_BONDS / r)]).T
#     f [:-1] += fa
#     f [1:] += -fa
#
#     return f
#
# def get_angles_gradient(mol):
#     f = np.zeros((mol.n_atoms,3))
#     for i in range(mol.n_atoms-2):
#         a= mol.coords[i]
#         b= mol.coords[i+1]
#         c= mol.coords[i+2]
#         ab = b-a
#         bc = c-b
#         theta = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))
#         h = -2*src.constants.K_ANGLES *(theta-src.constants.THETA0_ANGLES)
#         pa = np.cross(-ab, np.cross(-ab, bc))
#         pc = np.cross(-bc, np.cross(-ab, bc))
#         fa = (h/np.linalg.norm(ab))*(pa/np.linalg.norm(pa))
#         fc = (h/np.linalg.norm(bc))*(pc/np.linalg.norm(pc))
#         f[i] += fa
#         f[i+1] += -fa - fc
#         f[i+2] += fc
#     return f
#
# def get_torsions_gradient(mol):
#     f = np.zeros((mol.n_atoms,3))
#     for i in range(mol.n_atoms-3):
#         a= mol.coords[i]
#         b= mol.coords[i+1]
#         c= mol.coords[i+2]
#         d= mol.coords[i+3]
#
#         F = a-b
#         G = b-c
#         H = d-c
#         A = np.cross(F,G)
#         B = np.cross(H,G)
#
#         cos =  np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
#         sin =  np.dot(np.cross(B, A), G)/(np.linalg.norm(A)*np.linalg.norm(B)*np.linalg.norm(G))
#         phi = -np.arctan(sin/cos)
#         dU = - src.constants.K_TORSIONS*src.constants.N_TORSIONS*np.sin(src.constants.N_TORSIONS*phi - src.constants.DELTA_TORSIONS)
#
#         A2 = np.dot(A,A)
#         B2 = np.dot(B,B)
#         AG = (np.linalg.norm(G)/(A2))*A
#         AFG = (np.dot(F,G)/(np.linalg.norm(G)*A2))*A
#         BG = (np.linalg.norm(G)/(B2))*B
#         BHG = (np.dot(H,G)/(np.linalg.norm(G)*B2))*B
#
#         fa = -AG
#         fb = AG + AFG - BHG
#         fc = BHG - AFG - BG
#         fd = BG
#
#         f[i]   += dU*fa
#         f[i+1] += dU*fb
#         f[i+2] += dU*fc
#         f[i+3] += dU*fd
#
#     return f


################################################################################################
#    Automatic differentiation Gradient
################################################################################################

def get_autograd(mol):
    # print("GRADIENT COMPUTATION")
    F = np.zeros(mol.coords.shape)
    for i in range(mol.n_chain):
        F[mol.chain_id[i]:mol.chain_id[i+1]] += get_bonds_autograd(mol.get_chain(i))
        F[mol.chain_id[i]:mol.chain_id[i+1]] += get_angles_autograd(mol.get_chain(i))
        F[mol.chain_id[i]:mol.chain_id[i+1]] += get_torsions_autograd(mol.get_chain(i))
    return F

def get_bonds_autograd(x):
    grad_bonds = elementwise_grad(get_bonds_energy)
    return grad_bonds(x)

def get_angles_autograd(x):
    grad_angles = elementwise_grad(get_angles_energy)
    return grad_angles(x)

def get_torsions_autograd(x):
    grad_torsions = elementwise_grad(get_torsions_energy)
    return grad_torsions(x)


def get_autograd_NMA(mol, x, q, A):
    # print("GRADIENT COMPUTATION")
    Fx = np.zeros(x.shape)
    Fq = np.zeros(q.shape)

    for i in range(mol.n_chain):
        Fx_bonds, Fq_bonds = get_bonds_NMA_autograd(x[mol.chain_id[i]:mol.chain_id[i + 1]],
                                                       q, A[mol.chain_id[i]:mol.chain_id[i + 1]], mol.get_chain(i))
        Fx_angles, Fq_angles = get_angles_NMA_autograd(x[mol.chain_id[i]:mol.chain_id[i + 1]],
                                                       q, A[mol.chain_id[i]:mol.chain_id[i + 1]], mol.get_chain(i))
        Fx_torsions, Fq_torsions = get_torsions_NMA_autograd(x[mol.chain_id[i]:mol.chain_id[i + 1]],
                                                       q, A[mol.chain_id[i]:mol.chain_id[i + 1]], mol.get_chain(i))

        Fx[mol.chain_id[i]:mol.chain_id[i + 1]] = Fx_bonds + Fx_angles + Fx_torsions
        Fq += Fq_bonds + Fq_angles + Fq_torsions
    return Fx, Fq


def get_bonds_NMA_autograd(x, q, A, x0):
    grad_bonds = elementwise_grad(get_bonds_NMA_energy,(0,1))
    return grad_bonds(x, q, A, x0)

def get_angles_NMA_autograd(x, q, A, x0):
    grad_angles = elementwise_grad(get_angles_NMA_energy,(0,1))
    return grad_angles(x, q, A, x0)

def get_torsions_NMA_autograd(x, q, A, x0):
    grad_torsions = elementwise_grad(get_torsions_NMA_energy,(0,1))
    return grad_torsions(x, q, A, x0)

def get_bonds_NMA_energy(x, q, A, x0):
    coord = x0+ x+ npg.dot(q, A)
    r = npg.linalg.norm(coord[1:] - coord[:-1], axis=1)
    return npg.sum(src.constants.K_BONDS * npg.square(r - src.constants.R0_BONDS))

def get_angles_NMA_energy(x, q, A, x0):
    coord = x0 + x+ npg.dot(q, A)
    theta = npg.arccos(npg.sum((coord[:-2] - coord[1:-1]) * (coord[1:-1] - coord[2:]), axis=1)
                       / (npg.linalg.norm(coord[:-2] - coord[1:-1], axis=1) * npg.linalg.norm(
        coord[1:-1] - coord[2:], axis=1)))
    return npg.sum(src.constants.K_ANGLES * npg.square(theta - src.constants.THETA0_ANGLES))

def get_torsions_NMA_energy(x, q, A, x0):
    coord = x0 + x+ npg.dot(q, A)
    u1 = coord[1:-2] - coord[:-3]
    u2 = coord[2:-1] - coord[1:-2]
    u3 = coord[3:] - coord[2:-1]
    torsions = npg.arctan2(npg.linalg.norm(u2, axis=1) * npg.sum(u1 * npg.cross(u2, u3), axis=1),
                       npg.sum(npg.cross(u1, u2) * npg.cross(u2, u3), axis=1))
    return npg.sum(src.constants.K_TORSIONS*(1 + npg.cos(src.constants.N_TORSIONS*(torsions) - src.constants.DELTA_TORSIONS)))
################################################################################################
#    Internal Gradient
################################################################################################
# def get_gradient_i(mol):
#     print("GRADIENT COMPUTATION")
#     F_bonds = get_bonds_gradient_i(mol)
#     F_angles = get_angles_gradient_i(mol)
#     F_torsions = get_torsions_gradient_i(mol)
#     return F_bonds, F_angles, F_torsions
#
# def get_bonds_gradient_i(mol):
#     return 2*src.constants.K_BONDS*(mol.bonds - src.constants.R0_BONDS)
#
# def get_angles_gradient_i(mol):
#     return 2*src.constants.K_ANGLES*(mol.angles - src.constants.THETA0_ANGLES)
#
# def get_torsions_gradient_i(mol):
#     return -src.constants.K_TORSIONS*src.constants.N_TORSIONS*(np.sin(src.constants.N_TORSIONS*(mol.torsions) - src.constants.DELTA_TORSIONS))


################################################################################################
#    Kinetic Energy
################################################################################################
def get_kinetic_energy(velocities):
    return 1/2 * src.constants.CARBON_MASS * np.sum(np.square(velocities))