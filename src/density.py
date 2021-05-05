# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
# * IMPMC, UPMC Sorbonne University
# *
# **************************************************************************

import copy
import matplotlib.pyplot as plt
import numpy as np

import src.functions
import src.io
import src.viewers
from src.constants import FIT_VAR_LOCAL,FIT_VAR_GLOBAL, FIT_VAR_ROTATION, FIT_VAR_SHIFT



class Density:
    """
    Cryo EM Density
    """
    def __init__(self, data, voxel_size, sigma, cutoff):
        """
        Constructor
        :param data: numpy array containing pixels/voxels data
        :param voxel_size: voxels/pixel size
        :param sigma: TODO
        :param cutoff: TODO
        """
        self.data =data
        self.voxel_size = voxel_size
        self.sigma = sigma
        self.size = data.shape[0]
        self.cutoff = cutoff

    @classmethod
    def from_coords(cls, coord, size, sigma, voxel_size, cutoff):
        """
        Constructor from Cartesian coordinates
        :param coord: Cartesian coordinates
        :param size: number of pixel/voxel in a row for the density
        :param sigma: standard deviation of gaussian kernels
        :param voxel_size: voxel/pixel size
        :param cutoff: the distance from the atom beyond which the gaussian kernels are not integrated anymore
        """
        pass

    def rescale(self, density, method="normal"):
        """
        Rescale the density to fit an other density
        :param density: The other density to fit the scale
        :param method: normal or opt
        """
        if method=="normal":
            self.data =  ((self.data-self.data.mean())/self.data.std())*density.data.std() + density.data.mean()
        elif method =="opt":
            min1 = density.data.min()
            min2 = self.data.min()
            max1 = density.data.max()
            max2 = self.data.max()
            self.data = ((self.data - (min2 + min1))*(max1 - min1) )/ (max2 - min2)
        else:
            print("Undefined type")

    def compare_hist(self, density):
        """
        Compare histogram to an other density
        :param density: the other density
        """
        h1  = np.histogram(self.data.flatten(),bins=100)
        h2  = np.histogram(density.data.flatten(),bins=100)
        plt.figure()
        plt.hist(self.data.flatten(), 100,label="self")
        plt.hist(density.data.flatten(), 100,label="compared")
        plt.legend()

        plt.figure()
        plt.plot(h1[1][:-1] , np.cumsum(h1[0]), label="self")
        plt.plot(h2[1][:-1] , np.cumsum(h2[0]), label="compared")
        plt.legend()

    def show(self):
        """
        Show the density in matplotlib
        """
        pass

class Volume(Density):
    """
    Cryo EM density volume
    """

    @classmethod
    def from_coords(cls, coord, size, sigma, voxel_size, cutoff):
        vol,_ = pdb2vol(coord=coord, size=size, sigma=sigma, voxel_size=voxel_size, cutoff=cutoff)
        return cls(data=vol, voxel_size=voxel_size, sigma=sigma, cutoff=cutoff)

    @classmethod
    def from_file(cls, file, sigma, cutoff, voxel_size=None):
        data, vs = src.io.read_mrc(file)
        if voxel_size is None:
            voxel_size=vs
        return cls(data=data, voxel_size=voxel_size, sigma=sigma, cutoff=cutoff)

    def show(self):
        src.viewers.volume_viewer(self)

    def save_mrc(self, file):
        src.io.save_mrc(data=self.data, voxel_size=self.voxel_size, file=file)

    def resize(self, size):
        vol = np.zeros((size, size, size))
        low = size//2 - self.size//2
        high = size - low
        vol[low:high, low:high, low:high] = self.data
        self.data= vol
        self.size=size

class Image(Density):
    """
    Cryo EM density image
    """
    @classmethod
    def from_coords(cls, coord, size, sigma, voxel_size, cutoff):
        img,_ = pdb2img(coord=coord, size=size, sigma=sigma, voxel_size=voxel_size, cutoff=cutoff)
        return cls(data=img, voxel_size=voxel_size, sigma=sigma, cutoff=cutoff)

    def show(self):
        src.viewers.image_viewer(self)


def pdb2vol(coord, size, sigma, voxel_size, cutoff):
    vox, n_vox = src.functions.select_voxels(coord, size, voxel_size, cutoff)
    n_atoms = coord.shape[0]
    vol = np.zeros((size, size, size))
    expnt = np.zeros((n_atoms, n_vox, n_vox, n_vox))
    for i in range(n_atoms):
        mu = (np.mgrid[vox[i, 0]:vox[i, 0] + n_vox,
              vox[i, 1]:vox[i, 1] + n_vox,
              vox[i, 2]:vox[i, 2] + n_vox] - size / 2) * voxel_size
        x = np.repeat(coord[i], n_vox ** 3).reshape(3, n_vox, n_vox, n_vox)
        expnt[i] = np.exp(-np.square(np.linalg.norm(x - mu, axis=0)) / (2 * (sigma ** 2)))
        vol[vox[i, 0]:vox[i, 0] + n_vox,
        vox[i, 1]:vox[i, 1] + n_vox,
        vox[i, 2]:vox[i, 2] + n_vox] += expnt[i]
    return vol, expnt

def pdb2img(coord, size, sigma, voxel_size, cutoff):
    vox, n_pix = src.functions.select_voxels(coord, size, voxel_size, cutoff)
    pix = vox[:, :2]
    n_atoms = coord.shape[0]
    img = np.zeros((size, size))
    expnt = np.zeros((n_atoms, n_pix, n_pix))
    for i in range(n_atoms):
        mu = (np.mgrid[pix[i, 0]:pix[i, 0] + n_pix,
              pix[i, 1]:pix[i, 1] + n_pix] - size / 2) * voxel_size
        x = np.repeat(coord[i, :2], n_pix ** 2).reshape(2, n_pix, n_pix)
        expnt[i] = np.exp(-np.square(np.linalg.norm(x - mu, axis=0)) / (2 * (sigma ** 2)))
        img[pix[i, 0]:pix[i, 0] + n_pix,
        pix[i, 1]:pix[i, 1] + n_pix] += expnt[i]

    return img, expnt

def get_CC(map1, map2):
    return np.sum(map1*map2)/np.sqrt(np.sum(np.square(map1))*np.sum(np.square(map2)))

def get_LS(map1, map2):
    return np.linalg.norm(map1-map2)**2

def get_gradient_LS(mol, psim, pexp, params, **kwargs):
    """
    Compute the gradient of the lest squares between the density and a simultaed denisty
    :param mol: Mol used for simulting densitty
    :param psim: Simulated Density
    :param pexp: Experimental Density
    :param params: dictionnary of parameters to get the gradient (key= param name, value = param current value)
    :return: gradient of least squares for each parameters
    """
    coord = copy.copy(mol.coords)
    pdiff = psim - pexp.data


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
            expnt = kwargs["expnt"][i]
        else:
            expnt=np.exp(-np.square(np.linalg.norm(coord_grid - mu_grid, axis=0)) / (2 * (pexp.sigma ** 2)))
        expnt *= -(1 / (pexp.sigma ** 2)) * 2
        tmp = pdiff[vox[i, 0]:vox[i, 0] + n_vox,
                  vox[i, 1]:vox[i, 1] + n_vox,
                  vox[i, 2]:vox[i, 2] + n_vox] * expnt
        dpsim = np.sum((coord_grid - mu_grid) * np.array([tmp, tmp, tmp]), axis=(1, 2, 3))

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

def get_gradient_LS_img(mol, psim, pexp, params, **kwargs):
    """
    Compute the gradient of the least squares between the images
    :param mol: Mol used for simulting densitty
    :param psim: Simulated Density
    :param pexp: Experimental Density
    :param params: dictionnary of parameters to get the gradient (key= param name, value = param current value)
    :return: gradient of LS for each parameters
    """
    coord = copy.copy(mol.coords)
    pdiff = psim - pexp.data

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
    vox, n_pix = src.functions.select_voxels(coord, pexp.size, pexp.voxel_size, pexp.cutoff)
    pix = vox[:, :2]

    # perform gradient computation of all atoms
    for i in range(mol.n_atoms):
        mu_grid = (np.mgrid[pix[i, 0]:pix[i, 0] + n_pix,
                pix[i, 1]:pix[i, 1] + n_pix] - pexp.size / 2) * pexp.voxel_size
        coord_grid = np.repeat(coord[i, :2], n_pix ** 2).reshape(2, n_pix, n_pix)
        if "expnt" in kwargs:  # use a past
            expnt = kwargs["expnt"][i]
        else:
            expnt = np.exp(-np.square(np.linalg.norm(coord_grid - mu_grid, axis=0)) / (2 * (pexp.sigma ** 2)))
        tmp = 2 * pdiff[pix[i, 0]:pix[i, 0] + n_pix,
                      pix[i, 1]:pix[i, 1] + n_pix] * expnt
        dpsim = np.sum(-(1 / (pexp.sigma ** 2)) * (coord_grid - mu_grid) * np.array([tmp, tmp]), axis=(1, 2))

        # apply to parameters
        if FIT_VAR_SHIFT in params:
            res[FIT_VAR_SHIFT][:2] += dpsim
        if FIT_VAR_ROTATION in params:
            dR = src.functions.get_euler_grad(params[FIT_VAR_ROTATION], coord0[i])
            res[FIT_VAR_ROTATION] += np.dot(dR[:, :2], dpsim)
            dpsim *= (R[0] + R[1] + R[2])[:2]
        if FIT_VAR_LOCAL in params:
            res[FIT_VAR_LOCAL][i][:2] = dpsim
        if FIT_VAR_GLOBAL in params:
            res[FIT_VAR_GLOBAL] += np.dot(mol.normalModeVec[i, :, :2], dpsim)

    return res


def get_gradient_CC(mol, psim, pexp, params, **kwargs):
    """
    Compute the gradient of the CC between the density and a simultaed denisty
    :param mol: Mol used for simulting densitty
    :param psim: Simulated Density
    :param pexp: Experimental Density
    :param params: dictionnary of parameters to get the gradient (key= param name, value = param current value)
    :return: gradient of CC for each parameters
    """
    coord = copy.copy(mol.coords)

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

    psim_arr =psim
    pexp_arr =pexp.data
    psim2 = np.square(psim_arr)
    pexp2 = np.square(pexp_arr)
    spsim2 = np.sum(psim2)
    spexp2 = np.sum(pexp2)
    const1 = 1/np.sqrt(spsim2 * spexp2)
    const2 = np.sum(psim_arr*pexp_arr) / (np.sqrt(spexp2) * np.power(spsim2,3/2))

    # perform gradient computation of all atoms
    for i in range(mol.n_atoms):

        mu_grid = (np.mgrid[vox[i, 0]:vox[i, 0] + n_vox,
              vox[i, 1]:vox[i, 1] + n_vox,
              vox[i, 2]:vox[i, 2] + n_vox] - pexp.size / 2) * pexp.voxel_size
        coord_grid = np.repeat(coord[i], n_vox ** 3).reshape(3, n_vox, n_vox, n_vox)
        if "expnt" in kwargs: # use a past
            expnt = kwargs["expnt"][i]
        else:
            expnt= np.exp(-np.square(np.linalg.norm(coord_grid - mu_grid, axis=0)) / (2 * (pexp.sigma ** 2)))
        expnt *= 2* -(1 / (pexp.sigma ** 2))

        dpsim = (coord_grid - mu_grid) * np.array([expnt, expnt, expnt])

        term1 = psim_arr[vox[i, 0]:vox[i, 0] + n_vox,
                    vox[i, 1]:vox[i, 1] + n_vox,
                    vox[i, 2]:vox[i, 2] + n_vox] * dpsim
        term2 = pexp_arr[vox[i, 0]:vox[i, 0] + n_vox,
                    vox[i, 1]:vox[i, 1] + n_vox,
                    vox[i, 2]:vox[i, 2] + n_vox] * dpsim
        dcc = np.sum(term1, axis=(1, 2, 3)) * const1 - np.sum(term2, axis=(1, 2, 3)) * const2

        # apply to parameters
        if FIT_VAR_ROTATION in params:
            dR = src.functions.get_euler_grad(params[FIT_VAR_ROTATION], coord0[i])
            res[FIT_VAR_ROTATION] += np.dot(dR, dcc)
            dcc *= (R[0] + R[1]+ R[2])
        if FIT_VAR_LOCAL in params:
            res[FIT_VAR_LOCAL][i] = dcc
        if FIT_VAR_GLOBAL in params:
            res[FIT_VAR_GLOBAL] += np.dot(kwargs["normalModeVec"][i], dcc)
        if FIT_VAR_SHIFT in params:
            res[FIT_VAR_SHIFT] += dcc

    return res