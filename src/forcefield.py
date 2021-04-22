import autograd.numpy as npg
from autograd import elementwise_grad
import numpy as np
import copy
import sys
import time

from src.constants import *
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
        potentials = ["bonds", "angles", "dihedrals", "impropers", "vdw", "elec"]
    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    else:
        verbose=False
    U={}
    if verbose:
        print("Computing Potential energy ...")

    # Bonds energy
    if "bonds" in potentials:
        U["bonds"] = get_energy_bonds(coords, forcefield)

    # Angles energy
    if "angles" in potentials:
        U["angles"] = get_energy_angles(coords, forcefield)

    # Dihedrals energy
    if "dihedrals" in potentials:
        U["dihedrals"] = get_energy_dihedrals(coords, forcefield)

    # Impropers energy
    if "impropers" in potentials:
        U["impropers"] = get_energy_impropers(coords, forcefield)

    # Non bonded energy
    if "vdw" in potentials or "elec" in potentials:
        if "pairlist" in kwargs:
            pairlist = kwargs["pairlist"]
        else:
            if "cutoff" in kwargs:
                cutoff = kwargs["cutoff"]
            else:
                cutoff = 10.0
            pairlist = get_pairlist(coords, excluded_pairs=forcefield.excluded_pairs, cutoff=cutoff)
        invdist = get_invdist(coords, pairlist)

        if "vdw" in potentials:
            U["vdw"] = get_energy_vdw(invdist, pairlist, forcefield)

        if "elec" in potentials:
            U["elec"] = get_energy_elec(invdist, pairlist, forcefield)

    if verbose:
        for i in U:
            print("|-- "+i+" = " + str(round(U[i], 2)))

    U["total"] = np.sum([U[i] for i in U])
    if verbose:
        print("|== TOTAL = " + str(round(U["total"], 2)))

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
            coord += npg.dot(params[FIT_VAR_GLOBAL], kwargs["normalModeVec"])
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
        if "impropers" in kwargs["potentials"]:
            U += get_energy_impropers(coord, mol.forcefield)
        if "vdw" in kwargs["potentials"] or "elec" in kwargs["potentials"]:
            invdist = get_invdist(coord, kwargs["pairlist"])
            if "vdw" in kwargs["potentials"]:
                U += get_energy_vdw(invdist, kwargs["pairlist"], mol.forcefield)
            if "elec" in kwargs["potentials"]:
                U += get_energy_elec(invdist, kwargs["pairlist"], mol.forcefield)

        return U
    grad = elementwise_grad(get_energy_autograd, 0)

    F = grad(params, mol, **kwargs) # Get the derivative of the potential energy

    # EDIT TODO
    if "elec" in kwargs["potentials"]:
        if "local" in F:
            limit = 1000
            Fabs = np.linalg.norm(F["local"], axis=1)
            idx= np.where(Fabs > limit)[0]
            if idx.shape[0]>0:
                print(" ===============  LIMITER ACTIVATED  =================  ")
                F["local"][idx] =  (F["local"][idx].T * limit/Fabs[idx]).T
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

def get_energy_impropers(coord,forcefield):
    """
    Compute impropers potnetial
    :param coord: Cartesian coordinates (Agnstrom)
    :param forcefield: MoleculeForceField
    :return: impropers potential  (kcal * mol-1)
    """
    rji = coord[forcefield.impropers[:, 0]] - coord[forcefield.impropers[:, 1]]
    rjk = coord[forcefield.impropers[:, 2]] - coord[forcefield.impropers[:, 1]]
    rkj = -rjk
    rkl = coord[forcefield.impropers[:, 3]] - coord[forcefield.impropers[:, 2]]
    ra = npg.cross(rji, rjk)
    rb = npg.cross(rkj, rkl)
    psi = npg.arccos(npg.sum(ra*rb, axis=1)/ (npg.linalg.norm(ra, axis=1) * npg.linalg.norm(rb, axis=1)))
    return npg.sum(forcefield.Kpsi * (psi - forcefield.psi0*npg.pi/180)**2)

def get_excluded_pairs(forcefield):
    excluded_pairs = {}
    pairs = np.concatenate((forcefield.bonds, forcefield.angles[:,[0,2]]))
    pairs = np.concatenate((pairs, forcefield.dihedrals[:,[0,3]]))
    for i in pairs:
        if i[0] in excluded_pairs:
            excluded_pairs[i[0]].append(i[1])
        else:
            excluded_pairs[i[0]]= [i[1]]
        if i[1] in excluded_pairs:
            excluded_pairs[i[1]].append(i[0])
        else:
            excluded_pairs[i[1]]= [i[0]]
    for i in excluded_pairs:
        excluded_pairs[i] = np.array(excluded_pairs[i])
    return excluded_pairs

def get_pairlist(coord, excluded_pairs, cutoff=10.0, verbose=False):
    if verbose : print("Building pairlist ...")
    t=time.time()
    pairlist = []
    # invpairlist= []
    n_atoms=coord.shape[0]
    for i in range(n_atoms):
        dist_idx = np.setdiff1d(np.arange(i + 1, n_atoms), excluded_pairs[i])
        dist = np.linalg.norm(coord[dist_idx] - coord[i], axis=1)
        idx = dist_idx[np.where(dist < cutoff)[0] ]
        # invpairlist.append(idx)
        for j in idx:
            pairlist.append([i, j])
    pl_arr= np.array(pairlist)
    if verbose :
        print("Done ")
        print("\t Size : " + str(sys.getsizeof(pl_arr) / (8 * 1024)) + " kB")
        print("\t Time : " + str(time.time()-t) + " s")
    return pl_arr

def get_invdist(coord, pairlist):
    dist = npg.linalg.norm(coord[pairlist[:, 0]] - coord[pairlist[:, 1]], axis=1)
    return 1/dist

def get_energy_vdw(invdist, pairlist, forcefield):
    Rminij = forcefield.Rmin[pairlist[:, 0]] + forcefield.Rmin[pairlist[:, 1]]
    Epsij = npg.sqrt(forcefield.epsilon[pairlist[:, 0]] * forcefield.epsilon[pairlist[:, 1]])
    invdist6 = (Rminij * invdist) ** 6
    invdist12 = invdist6 ** 2
    return npg.sum(Epsij * (invdist12 - 2 * invdist6)) *2/10

def get_energy_elec(invdist, pairlist, forcefield):
    # Electrostatics
    U = npg.sum( forcefield.charge[pairlist[:, 0]] * forcefield.charge[pairlist[:, 1]] *invdist)
    return U * (ELEMENTARY_CHARGE) ** 2 * AVOGADRO_CONST / \
     (VACUUM_PERMITTIVITY * WATER_RELATIVE_PERMIT * ANGSTROM_TO_METER * KCAL_TO_JOULE)*2

def get_gradient_RMSD(mol, psim, pexp, params, **kwargs):
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
        res[FIT_VAR_GLOBAL] = np.zeros(kwargs["normalModeVec"].shape[1])
        coord += np.dot(params[FIT_VAR_GLOBAL], kwargs["normalModeVec"])
    if FIT_VAR_ROTATION in params:
        res[FIT_VAR_ROTATION] = np.zeros(3)
        R = src.functions.generate_euler_matrix(angles=params[FIT_VAR_ROTATION])
        coord0 = coord
        coord = np.dot(R, coord.T).T
    if FIT_VAR_SHIFT in params:
        res[FIT_VAR_SHIFT] = np.zeros(3)
        coord += params[FIT_VAR_SHIFT]

    # Select impacted voxels
    vox, n_vox = src.functions.select_voxels(coord, pexp.size, pexp.voxel_size, pexp.cutoff)

    # perform gradient computation of all atoms
    for i in range(mol.n_atoms):
        mu_grid = (np.mgrid[vox[i, 0]:vox[i, 0] + n_vox,
              vox[i, 1]:vox[i, 1] + n_vox,
              vox[i, 2]:vox[i, 2] + n_vox] - pexp.size / 2) * pexp.voxel_size
        coord_grid = np.repeat(coord[i], n_vox ** 3).reshape(3, n_vox, n_vox, n_vox)
        if "expnt" in kwargs: # use a past
            e = kwargs["expnt"][i]
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
            res[FIT_VAR_GLOBAL] += np.dot(kwargs["normalModeVec"][i], dpsim)
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
    #     vox, n_pix = src.functions.select_voxels(coord, self.size, self.voxel_size, self.cutoff)
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


