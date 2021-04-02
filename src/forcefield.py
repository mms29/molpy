import autograd.numpy as npg
from autograd import elementwise_grad
import numpy as np
import copy

from src.constants import FIT_VAR_LOCAL,FIT_VAR_GLOBAL, FIT_VAR_ROTATION, FIT_VAR_SHIFT
import src.functions


def get_energy(coords, forcefield, **kwargs):
    """
    Compute the potential energy
    :param coords: np array N*3 (the Cartesian coordinates(Angstrom))
    :param forcefield: MoleculeForceField object
    :return: Potential energy (kcal * mol-1)
    """
    if "potentials" in kwargs:
        potentials= kwargs["potentials"]
    else:
        potentials = ["bonds", "angles", "dihedrals", "vdw", "elec"]
    U=0
    if "verbose" in kwargs:
        print("Computing Potential energy ...")

    # Bonds energy
    if "bonds" in potentials:
        U_bonds = get_energy_bonds(coords, forcefield)
        U+=U_bonds
        if "verbose" in kwargs:
            print("|-- Bonds = " + str(round(U_bonds, 2)))

    # Angles energy
    if "angles" in potentials:
        U_angles = get_energy_angles(coords, forcefield)
        U+=U_angles
        if "verbose" in kwargs:
            print("|-- Angles = " + str(round(U_angles, 2)))

    # Dihedrals energy
    if "dihedrals" in potentials:
        U_dihedrals = get_energy_dihedrals(coords, forcefield)
        U+=U_dihedrals
        if "verbose" in kwargs:
            print("|-- Dihedrals = " + str(round(U_dihedrals, 2)))

    # Non bonded energy
    if "vdw" in potentials or "elec" in potentials:
        if "pairlist" in kwargs:
            pairlist = kwargs["pairlist"]
        else:
            pairlist = get_pairlist(coords, 10.0)
        invdist = get_invdist(coords, pairlist)

        if "vdw" in potentials:
            U_vdw = get_energy_vdw(invdist, pairlist, forcefield)
            U += U_vdw
            if "verbose" in kwargs:
                print("|-- Van der Waals = " + str(round(U_vdw, 2)))

        if "elec" in potentials:
            U_elec = get_energy_elec(invdist, pairlist, forcefield)
            U += U_elec
            if "verbose" in kwargs:
                print("|-- Electrostatics = " + str(round(U_elec, 2)))

    if "verbose" in kwargs:
        print("|== TOTAL = " + str(round(U, 2)))

    return U

def get_autograd(params, mol, **kwargs):
    """
    Compute the gradient of the potential energy by automatic differentiation
    :param params: dictionary with keys are the name of the parameters and values their values
    :param mol: initial Molecule
    :return: gradient for each parameter  (kcal * mol-1 * A)
    """
    def get_energy_autograd(params, mol, **kwargs):
        """
        Energy function for automatic differentiation
        """
        coord = npg.array(mol.coords)
        if FIT_VAR_LOCAL in params:
            coord += params[FIT_VAR_LOCAL]
        if FIT_VAR_GLOBAL in params:
            coord += npg.dot(params[FIT_VAR_GLOBAL], mol.normalModeVec)
        if FIT_VAR_ROTATION in params:
            coord = npg.dot(src.functions.generate_euler_matrix(params[FIT_VAR_ROTATION]), coord.T).T
        if FIT_VAR_SHIFT in params:
            coord += params[FIT_VAR_SHIFT]

        U=0

        if "bonds" in kwargs["potentials"]:
            U += get_energy_bonds(coord, mol.forcefield)
        if "angles" in kwargs["potentials"]:
            U += get_energy_angles(coord, mol.forcefield)
        if "dihedrals" in kwargs["potentials"]:
            U += get_energy_dihedrals(coord, mol.forcefield)
        if "vdw" in kwargs["potentials"] or "elec" in kwargs["potentials"]:
            invdist = get_invdist(coord, kwargs["pairlist"])
            if "vdw" in kwargs["potentials"]:
                U += get_energy_vdw(invdist, kwargs["pairlist"], mol.forcefield)
            if "elec" in kwargs["potentials"]:
                U += get_energy_elec(invdist, kwargs["pairlist"], mol.forcefield)

        return U
    grad = elementwise_grad(get_energy_autograd, 0)

    F = grad(params, mol, **kwargs) # Get the derivative of the potential energy
    return F

def get_energy_bonds(coord, forcefield):
    """
    Compute bonds potential
    :param coord: Cartesian coordinates (Angstrom)
    :param forcefield: MoleculeForceField
    :return: bonds potential  (kcal * mol-1)
    """
    r = npg.linalg.norm(coord[forcefield.bonds[:, 0]] - coord[forcefield.bonds[:, 1]], axis=1)
    return npg.sum(forcefield.Kb * npg.square(r - forcefield.b0))

def get_energy_angles(coord, forcefield):
    """
    Compute angles potnetial
    :param coord: Cartesian coordinates (Angstrom)
    :param forcefield: MoleculeForceField
    :return: angles potential (kcal * mol-1)
    """
    a1 = coord[forcefield.angles[:, 0]]
    a2 = coord[forcefield.angles[:, 1]]
    a3 = coord[forcefield.angles[:, 2]]
    theta = -npg.arccos(npg.sum((a1 - a2) * (a2 - a3), axis=1)
                       / (npg.linalg.norm(a1 - a2, axis=1) * npg.linalg.norm(a2- a3, axis=1))) + npg.pi
    return npg.sum(forcefield.KTheta * npg.square(theta - (forcefield.Theta0*npg.pi/180)))

def get_energy_dihedrals(coord,forcefield):
    """
    Compute dihedrals potnetial
    :param coord: Cartesian coordinates (Agnstrom)
    :param forcefield: MoleculeForceField
    :return: dihedrals potential  (kcal * mol-1)
    """
    u1 = coord[forcefield.dihedrals[:, 1]] - coord[forcefield.dihedrals[:, 0]]
    u2 = coord[forcefield.dihedrals[:, 2]] - coord[forcefield.dihedrals[:, 1]]
    u3 = coord[forcefield.dihedrals[:, 3]] - coord[forcefield.dihedrals[:, 2]]
    torsions = npg.arctan2(npg.linalg.norm(u2, axis=1) * npg.sum(u1 * npg.cross(u2, u3), axis=1),
                           npg.sum(npg.cross(u1, u2) * npg.cross(u2, u3), axis=1))
    return npg.sum(forcefield.Kchi * (1 + npg.cos(forcefield.n * (torsions) - (forcefield.delta*npg.pi/180))))


def get_pairlist(coord,cutoff=10.0):
    pairlist = []
    for i in range(coord.shape[0]):
        dist = np.linalg.norm(coord[i + 1:] - coord[i], axis=1)
        idx = np.where(dist < cutoff)[0] + i + 1
        for j in idx:
            pairlist.append([i, j])
    return np.array(pairlist)

def get_invdist(coord, pairlist):
    dist = npg.linalg.norm(coord[pairlist[:, 0]] - coord[pairlist[:, 1]], axis=1)
    return 1/dist

def get_energy_vdw(invdist, pairlist, forcefield):
    Rminij = forcefield.Rmin[pairlist[:, 0]] + forcefield.Rmin[pairlist[:, 1]]
    Epsij = npg.sqrt(forcefield.epsilon[pairlist[:, 0]] * forcefield.epsilon[pairlist[:, 1]])
    invdist6 = (Rminij * invdist) ** 6
    invdist12 = invdist6 ** 2
    return npg.sum(Epsij * (invdist12 - 2 * invdist6)) *0.00004

def get_energy_elec(invdist, pairlist, forcefield):
    # Electrostatics
    eps0 = 0.0027865
    return npg.sum(1 / (4 * npg.pi *eps0) * forcefield.charge[pairlist[:, 0]] * forcefield.charge[pairlist[:, 1]] *invdist)

def get_energy_nonbonded(coord, pairlist, forcefield):
    invdist = get_invdist(coord, pairlist)
    U_vdw = get_energy_vdw(invdist, pairlist, forcefield)
    U_elec = get_energy_elec(invdist, pairlist, forcefield)
    return U_vdw+ U_elec


def get_gradient_RMSD(mol, psim, pexp, params, expnt=None):
    """
    Compute the gradient of the RMSD between the density and a simultaed denisty
    :param mol: Mol used for simulting densitty
    :param psim: Simulated Density
    :param pexp: Experimental Density
    :param params: dictionnary of parameters to get the gradient (key= param name, value = param current value)
    :return: gradient of RMSD for each parameters
    """
    coord = copy.copy(mol.coords)
    pdiff = psim.data - pexp.data


    # Forward model
    res = {}
    if FIT_VAR_LOCAL in params:
        res[FIT_VAR_LOCAL] = np.zeros(coord.shape)
        coord += params[FIT_VAR_LOCAL]
    if FIT_VAR_GLOBAL in params:
        res[FIT_VAR_GLOBAL] = np.zeros(mol.normalModeVec.shape[1])
        coord += np.dot(params[FIT_VAR_GLOBAL], mol.normalModeVec)
    if FIT_VAR_ROTATION in params:
        res[FIT_VAR_ROTATION] = np.zeros(3)
        R = src.functions.generate_euler_matrix(angles=params[FIT_VAR_ROTATION])
        coord0 = coord
        coord = np.dot(R, coord.T).T
    if FIT_VAR_SHIFT in params:
        res[FIT_VAR_SHIFT] = np.zeros(3)
        coord += params[FIT_VAR_SHIFT]

    # Select impacted voxels
    vox, n_vox = src.functions.select_voxels(coord, pexp.size, pexp.voxel_size, pexp.threshold)

    # perform gradient computation of all atoms
    for i in range(mol.n_atoms):
        mu_grid = (np.mgrid[vox[i, 0]:vox[i, 0] + n_vox,
              vox[i, 1]:vox[i, 1] + n_vox,
              vox[i, 2]:vox[i, 2] + n_vox] - pexp.size / 2) * pexp.voxel_size
        coord_grid = np.repeat(coord[i], n_vox ** 3).reshape(3, n_vox, n_vox, n_vox)
        if expnt is not None: # use a past
            e = expnt[i]
        else:
            e=np.exp(-np.square(np.linalg.norm(coord_grid - mu_grid, axis=0)) / (2 * (pexp.sigma ** 2)))
        tmp = 2 * pdiff[vox[i, 0]:vox[i, 0] + n_vox,
                  vox[i, 1]:vox[i, 1] + n_vox,
                  vox[i, 2]:vox[i, 2] + n_vox]*e
        dpsim = np.sum(-(1 / (pexp.sigma ** 2)) * (coord_grid - mu_grid) * np.array([tmp, tmp, tmp]), axis=(1, 2, 3))

        # apply to parameters
        if FIT_VAR_ROTATION in params:
            dR = src.functions.get_euler_grad(params[FIT_VAR_ROTATION], coord0[i])
            res[FIT_VAR_ROTATION] += np.dot(dR, dpsim)
            dpsim *= (R[0] + R[1]+ R[2])
        if FIT_VAR_LOCAL in params:
            res[FIT_VAR_LOCAL][i] = dpsim
        if FIT_VAR_GLOBAL in params:
            res[FIT_VAR_GLOBAL] += np.dot(mol.normalModeVec[i], dpsim)
        if FIT_VAR_SHIFT in params:
            res[FIT_VAR_SHIFT] += dpsim

    return res

    #
    # def get_gradient_RMSD(self, mol, psim, params):
    #     coord = copy.copy(mol.coords)
    #     pdiff = psim - self.data
    #
    #     res = {}
    #     if FIT_VAR_LOCAL in params:
    #         res[FIT_VAR_LOCAL] = np.zeros(coord.shape)
    #         coord += params[FIT_VAR_LOCAL]
    #     if FIT_VAR_GLOBAL in params:
    #         res[FIT_VAR_GLOBAL] = np.zeros(mol.normalModeVec.shape[1])
    #         coord += np.dot(params[FIT_VAR_GLOBAL], mol.normalModeVec)
    #     if FIT_VAR_ROTATION in params:
    #         res[FIT_VAR_ROTATION] = np.zeros(3)
    #         R = src.functions.generate_euler_matrix(angles=params[FIT_VAR_ROTATION])
    #         coord0 = coord
    #         coord = np.dot(R, coord.T).T
    #     if FIT_VAR_SHIFT in params:
    #         res[FIT_VAR_SHIFT] = np.zeros(3)
    #         coord += params[FIT_VAR_SHIFT]
    #
    #     vox, n_pix = src.functions.select_voxels(coord, self.size, self.voxel_size, self.threshold)
    #     pix = vox[:, :2]
    #
    #     for i in range(mol.n_atoms):
    #         mu_grid = (np.mgrid[pix[i, 0]:pix[i, 0] + n_pix,
    #               pix[i, 1]:pix[i, 1] + n_pix] - self.size / 2) * self.voxel_size
    #         coord_grid = np.repeat(coord[i, :2], n_pix ** 2).reshape(2, n_pix, n_pix)
    #         tmp = 2 * pdiff[pix[i, 0]:pix[i, 0] + n_pix,
    #                   pix[i, 1]:pix[i, 1] + n_pix] * np.exp(
    #             -np.square(np.linalg.norm(coord_grid - mu_grid, axis=0)) / (2 * (self.sigma ** 2)))
    #         dpsim = np.sum(-(1 / (self.sigma ** 2)) * (coord_grid - mu_grid) * np.array([tmp, tmp]), axis=(1, 2))
    #
    #
    #         if FIT_VAR_ROTATION in params:
    #             dR = src.functions.get_euler_grad(params[FIT_VAR_ROTATION], coord0[i])
    #             res[FIT_VAR_ROTATION] += np.dot(dR[:,:2], dpsim)
    #             dpsim *= (R[0] + R[1]+ R[2])[:2]
    #         if FIT_VAR_LOCAL in params:
    #             res[FIT_VAR_LOCAL][i] = dpsim
    #         if FIT_VAR_GLOBAL in params:
    #             res[FIT_VAR_GLOBAL] += np.dot(mol.normalModeVec[i, :, :2], dpsim)
    #         if FIT_VAR_SHIFT in params:
    #             res[FIT_VAR_SHIFT][:2] += dpsim
    #
    #     return res


