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
    def __init__(self, data, voxel_size, sigma, threshold):
        """
        Constructor
        :param data: numpy array containing pixels/voxels data
        :param voxel_size: voxels/pixel size
        :param sigma: TODO
        :param threshold: TODO
        """
        self.data =data
        self.voxel_size = voxel_size
        self.sigma = sigma
        self.size = data.shape[0]
        self.threshold = threshold

    @classmethod
    def from_coords(cls, coord, size, sigma, voxel_size, threshold):
        """
        Constructor from Cartesian coordinates
        :param coord: Cartesian coordinates
        :param size: number of pixel/voxel in a row for the density
        :param sigma: standard deviation of gaussian kernels
        :param voxel_size: voxel/pixel size
        :param threshold: the distance from the atom beyond which the gaussian kernels are not integrated anymore
        """
        pass

    def get_gradient_RMSD(self, mol, psim, params):
        """
        Compute the gradient of the RMSD between the density and a simultaed denisty
        :param mol: Mol used for simulting densitty
        :param psim: Simulated Density
        :param params: parameters to derive
        :return: gradient of RMSD for each parameters
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
    def from_coords(cls, coord, size, sigma, voxel_size, threshold):
        vox, n_vox = src.functions.select_voxels(coord, size, voxel_size, threshold)
        n_atoms = coord.shape[0]
        vol = np.zeros((size, size, size))
        for i in range(n_atoms):
            mu = (np.mgrid[vox[i, 0]:vox[i, 0] + n_vox,
                  vox[i, 1]:vox[i, 1] + n_vox,
                  vox[i, 2]:vox[i, 2] + n_vox] - size / 2) * voxel_size
            x = np.repeat(coord[i], n_vox ** 3).reshape(3, n_vox, n_vox, n_vox)
            vol[vox[i, 0]:vox[i, 0] + n_vox,
            vox[i, 1]:vox[i, 1] + n_vox,
            vox[i, 2]:vox[i, 2] + n_vox] += np.exp(-np.square(np.linalg.norm(x - mu, axis=0)) / (2 * (sigma ** 2)))

        return cls(data=vol, voxel_size=voxel_size, sigma=sigma, threshold=threshold)

    @classmethod
    def from_file(cls, file, sigma, threshold, voxel_size=None):
        data, vs = src.io.read_mrc(file)
        if voxel_size is None:
            voxel_size=vs
        return cls(data=data, voxel_size=voxel_size, sigma=sigma, threshold=threshold)

    def get_gradient_RMSD(self, mol, psim, params):
        coord = copy.copy(mol.coords)
        vox, n_vox = src.functions.select_voxels(coord, self.size, self.voxel_size, self.threshold)
        pdiff = psim - self.data

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

        for i in range(mol.n_atoms):
            mu_grid = (np.mgrid[vox[i, 0]:vox[i, 0] + n_vox,
                  vox[i, 1]:vox[i, 1] + n_vox,
                  vox[i, 2]:vox[i, 2] + n_vox] - self.size / 2) * self.voxel_size
            coord_grid = np.repeat(coord[i], n_vox ** 3).reshape(3, n_vox, n_vox, n_vox)
            tmp = 2 * pdiff[vox[i, 0]:vox[i, 0] + n_vox,
                      vox[i, 1]:vox[i, 1] + n_vox,
                      vox[i, 2]:vox[i, 2] + n_vox] * np.exp(
                -np.square(np.linalg.norm(coord_grid - mu_grid, axis=0)) / (2 * (self.sigma ** 2)))
            dpsim = np.sum(-(1 / (self.sigma ** 2)) * (coord_grid - mu_grid) * np.array([tmp, tmp, tmp]), axis=(1, 2, 3))


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

    def show(self):
        src.viewers.volume_viewer(self)

    def save_mrc(self, file):
        src.io.save_mrc(data=self.data, voxel_size=self.voxel_size, file=file)

class Image(Density):
    """
    Cryo EM density image
    """

    @classmethod
    def from_coords(cls, coord, size, sigma, voxel_size, threshold):
        vox, n_pix = src.functions.select_voxels(coord, size, voxel_size, threshold)
        pix = vox[:, :2]
        n_atoms = coord.shape[0]
        img = np.zeros((size, size))
        for i in range(n_atoms):
            mu = (np.mgrid[pix[i, 0]:pix[i, 0] + n_pix,
                  pix[i, 1]:pix[i, 1] + n_pix] - size / 2) * voxel_size
            x = np.repeat(coord[i, :2], n_pix ** 2).reshape(2, n_pix, n_pix)
            img[pix[i, 0]:pix[i, 0] + n_pix,
            pix[i, 1]:pix[i, 1] + n_pix] += np.exp(-np.square(np.linalg.norm(x - mu, axis=0)) / (2 * (sigma ** 2)))

        return cls(data=img, voxel_size=voxel_size, sigma=sigma, threshold=threshold)

    def get_gradient_RMSD(self, mol, psim, params):
        coord = copy.copy(mol.coords)
        vox, n_pix = src.functions.select_voxels(coord, self.size, self.voxel_size, self.threshold)
        pix = vox[:, :2]
        pdiff = psim - self.data

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

        for i in range(mol.n_atoms):
            mu_grid = (np.mgrid[pix[i, 0]:pix[i, 0] + n_pix,
                  pix[i, 1]:pix[i, 1] + n_pix] - self.size / 2) * self.voxel_size
            coord_grid = np.repeat(coord[i, :2], n_pix ** 2).reshape(2, n_pix, n_pix)
            tmp = 2 * pdiff[pix[i, 0]:pix[i, 0] + n_pix,
                      pix[i, 1]:pix[i, 1] + n_pix] * np.exp(
                -np.square(np.linalg.norm(coord_grid - mu_grid, axis=0)) / (2 * (self.sigma ** 2)))
            dpsim = np.sum(-(1 / (self.sigma ** 2)) * (coord_grid - mu_grid) * np.array([tmp, tmp]), axis=(1, 2))


            if FIT_VAR_ROTATION in params:
                dR = src.functions.get_euler_grad(params[FIT_VAR_ROTATION], coord0[i])
                res[FIT_VAR_ROTATION] += np.dot(dR[:,:2], dpsim)
                dpsim *= (R[0] + R[1]+ R[2])[:2]
            if FIT_VAR_LOCAL in params:
                res[FIT_VAR_LOCAL][i] = dpsim
            if FIT_VAR_GLOBAL in params:
                res[FIT_VAR_GLOBAL] += np.dot(mol.modes[i, :, :2], dpsim)
            if FIT_VAR_SHIFT in params:
                res[FIT_VAR_SHIFT][:2] += dpsim

        return res

    def show(self):
        src.viewers.image_viewer(self)