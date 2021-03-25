import autograd.numpy as npg
from autograd import elementwise_grad
import numpy as np
import copy

from src.constants import FIT_VAR_LOCAL,FIT_VAR_GLOBAL, FIT_VAR_ROTATION, FIT_VAR_SHIFT
import src.functions


def get_energy(coord, molstr, molprm, verbose=True):
    """
    Compute the potential energy
    :param coord: the Cartesian coordinates numpy array N*3 (Angstrom)
    :param molstr: Moleculestructure object
    :param molprm: MoleculeForceFieldPrm object
    :param verbose: verbose level
    :return: Potential energy (kcal * mol-1)
    """
    U_bonds = get_energy_bonds(coord, molstr.bonds, molprm)
    U_angles = get_energy_angles(coord, molstr.angles, molprm)
    U_dihedrals = get_energy_dihedrals(coord, molstr.dihedrals, molprm)
    U_total = U_bonds + U_angles + U_dihedrals

    if verbose:
        print("|-- Bonds = " + str(round(U_bonds, 2)))
        print("|-- Angles = " + str(round(U_angles, 2)))
        print("|-- Dihedrals = " + str(round(U_dihedrals, 2)))
        # print("|-- Van der Waals = " + str(round(U_torsions, 2)))
        print("|== TOTAL = " + str(round(U_total, 2)))

    return U_total

def get_autograd(params, mol):
    """
    Compute the gradient of the potential energy by automatic differentiation
    :param params: dictionary with keys are the name of the parameters and values their values
    :param mol: initial Molecule
    :return: gradient for each parameter  (kcal * mol-1 * A)
    """
    def get_energy_autograd(params, mol):
        """
        Energy function for automatic differentiation
        """
        coord = npg.array(mol.coords)
        if FIT_VAR_LOCAL in params:
            coord += params[FIT_VAR_LOCAL]
        if FIT_VAR_GLOBAL in params:
            coord += npg.dot(params[FIT_VAR_GLOBAL], mol.modes)
        if FIT_VAR_ROTATION in params:
            coord = npg.dot(src.functions.generate_euler_matrix(params[FIT_VAR_ROTATION]), coord.T).T
        if FIT_VAR_SHIFT in params:
            coord += params[FIT_VAR_SHIFT]

        U_bonds = get_energy_bonds(coord, mol.psf.bonds, mol.prm)
        U_angles = get_energy_angles(coord, mol.psf.angles, mol.prm)
        U_dihedrals = get_energy_dihedrals(coord, mol.psf.dihedrals, mol.prm)

        return (U_bonds + U_angles + U_dihedrals)
    grad = elementwise_grad(get_energy_autograd, 0)

    F = grad(params, mol) # Get the derivative of the potential energy
    return F

def get_energy_bonds(coord, bonds, prm):
    """
    Compute bonds potential
    :param coord: Cartesian coordinates (Angstrom)
    :param bonds: bonds index
    :param prm: MoleculeForceFieldPrm
    :return: bonds potential  (kcal * mol-1)
    """
    r = npg.linalg.norm(coord[bonds[:, 0]] - coord[bonds[:, 1]], axis=1)
    return npg.sum(prm.Kb * npg.square(r - prm.b0))

def get_energy_angles(coord, angles, prm):
    """
    Compute angles potnetial
    :param coord: Cartesian coordinates (Angstrom)
    :param angles: angles index
    :param prm: MoleculeForceFieldPrm
    :return: angles potential (kcal * mol-1)
    """
    a1 = coord[angles[:, 0]]
    a2 = coord[angles[:, 1]]
    a3 = coord[angles[:, 2]]
    theta = -npg.arccos(npg.sum((a1 - a2) * (a2 - a3), axis=1)
                       / (npg.linalg.norm(a1 - a2, axis=1) * npg.linalg.norm(a2- a3, axis=1))) + npg.pi
    return npg.sum(prm.KTheta * npg.square(theta - (prm.Theta0*npg.pi/180)))

def get_energy_dihedrals(coord, dihedrals, prm):
    """
    Compute dihedrals potnetial
    :param coord: Cartesian coordinates (Agnstrom)
    :param dihedrals: dihedrals index
    :param prm: MoleculeForceFieldPrm
    :return: dihedrals potential  (kcal * mol-1)
    """
    u1 = coord[dihedrals[:, 1]] - coord[dihedrals[:, 0]]
    u2 = coord[dihedrals[:, 2]] - coord[dihedrals[:, 1]]
    u3 = coord[dihedrals[:, 3]] - coord[dihedrals[:, 2]]
    torsions = npg.arctan2(npg.linalg.norm(u2, axis=1) * npg.sum(u1 * npg.cross(u2, u3), axis=1),
                           npg.sum(npg.cross(u1, u2) * npg.cross(u2, u3), axis=1))
    return npg.sum(prm.Kchi * (1 + npg.cos(prm.n * (torsions) - (prm.delta*npg.pi/180))))


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

    res = {}
    if FIT_VAR_LOCAL in params:
        res[FIT_VAR_LOCAL] = np.zeros(coord.shape)
        coord += params[FIT_VAR_LOCAL]
    if FIT_VAR_GLOBAL in params:
        res[FIT_VAR_GLOBAL] = np.zeros(mol.modes.shape[1])
        coord += np.dot(params[FIT_VAR_GLOBAL], mol.modes)
    if FIT_VAR_ROTATION in params:
        res[FIT_VAR_ROTATION] = np.zeros(3)
        R = src.functions.generate_euler_matrix(angles=params[FIT_VAR_ROTATION])
        coord0 = coord
        coord = np.dot(R, coord.T).T
    if FIT_VAR_SHIFT in params:
        res[FIT_VAR_SHIFT] = np.zeros(3)
        coord += params[FIT_VAR_SHIFT]

    vox, n_vox = src.functions.select_voxels(coord, pexp.size, pexp.voxel_size, pexp.threshold)

    for i in range(mol.n_atoms):
        mu_grid = (np.mgrid[vox[i, 0]:vox[i, 0] + n_vox,
              vox[i, 1]:vox[i, 1] + n_vox,
              vox[i, 2]:vox[i, 2] + n_vox] - pexp.size / 2) * pexp.voxel_size
        coord_grid = np.repeat(coord[i], n_vox ** 3).reshape(3, n_vox, n_vox, n_vox)
        if expnt is not None:
            e = expnt[i]
        else:
            e=np.exp(-np.square(np.linalg.norm(coord_grid - mu_grid, axis=0)) / (2 * (pexp.sigma ** 2)))
        tmp = 2 * pdiff[vox[i, 0]:vox[i, 0] + n_vox,
                  vox[i, 1]:vox[i, 1] + n_vox,
                  vox[i, 2]:vox[i, 2] + n_vox]*e
        dpsim = np.sum(-(1 / (pexp.sigma ** 2)) * (coord_grid - mu_grid) * np.array([tmp, tmp, tmp]), axis=(1, 2, 3))


        if FIT_VAR_ROTATION in params:
            dR = src.functions.get_euler_grad(params[FIT_VAR_ROTATION], coord0[i])
            res[FIT_VAR_ROTATION] += np.dot(dR, dpsim)
            dpsim *= (R[0] + R[1]+ R[2])
        if FIT_VAR_LOCAL in params:
            res[FIT_VAR_LOCAL][i] = dpsim
        if FIT_VAR_GLOBAL in params:
            res[FIT_VAR_GLOBAL] += np.dot(mol.modes[i], dpsim)
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
    #         res[FIT_VAR_GLOBAL] = np.zeros(mol.modes.shape[1])
    #         coord += np.dot(params[FIT_VAR_GLOBAL], mol.modes)
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
    #             res[FIT_VAR_GLOBAL] += np.dot(mol.modes[i, :, :2], dpsim)
    #         if FIT_VAR_SHIFT in params:
    #             res[FIT_VAR_SHIFT][:2] += dpsim
    #
    #     return res


