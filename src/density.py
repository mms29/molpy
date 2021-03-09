import numpy as np
import src.viewers
import src.functions
import copy
import src.io
import matplotlib.pyplot as plt

class Density:
    def __init__(self, data, voxel_size, sigma, threshold):
        self.data =data
        self.voxel_size = voxel_size
        self.sigma = sigma
        self.size = data.shape[0]
        self.threshold = threshold

    def get_gradient_RMSD(self, mol, psim, params):
        pass

    @classmethod
    def from_coords(cls, coord, size, sigma, voxel_size, threshold):
        pass

    @classmethod
    def from_file(cls, file):
        pass

    def show(self):
        pass

    def rescale(self, density):
        self.data =  ((self.data-self.data.mean())/self.data.std())*density.data.std() + density.data.mean()

    def rescale_opt(self, density):
        min1 = density.data.min()
        min2 = self.data.min()
        max1 = density.data.max()
        max2 = self.data.max()
        self.data = ((self.data - (min2 + min1))*(max1 - min1) )/ (max2 - min2)

    def compare_hist(self, density):
        h1  = np.histogram(self.data.flatten(),bins=100)
        h2  = np.histogram(density.data.flatten(),bins=100)
        plt.figure()
        plt.plot(h1[1][:-1] , np.cumsum(h1[0]), label="self")
        plt.plot(h2[1][:-1] , np.cumsum(h2[0]), label="compared")
        plt.legend()

class Volume(Density):

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
        # NB : coord = x0 + q*A +x
        coord = copy.copy(mol.coords)
        vox, n_vox = src.functions.select_voxels(coord, self.size, self.voxel_size, self.threshold)
        pdiff = psim - self.data

        res = {}
        if "x" in params:
            res["x"] = np.zeros(coord.shape)
            coord += params["x"]
        if "q" in params:
            res["q"] = np.zeros(mol.modes.shape[1])
            coord += np.dot(params["q"], mol.modes)
        if "angles" in params:
            res["angles"] = np.zeros(3)
            R = src.functions.generate_euler_matrix(angles=params["angles"])
            coord0 = coord
            coord = np.dot(R, coord.T).T
        if "shift" in params:
            res["shift"] = np.zeros(3)
            coord += params["shift"]

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
            if "angles" in params:
                dR = src.functions.get_euler_grad(params["angles"], coord0[i])
                res["angles"] += np.dot(dR, dpsim)
                dpsim *= (R[0] + R[1]+ R[2])
            if "x" in params:
                res["x"][i] = dpsim
            if "q" in params:
                res["q"] += np.dot(mol.modes[i], dpsim)
            if "shift" in params:
                res["shift"] += dpsim

        return res

    def show(self):
        src.viewers.density_viewer(self)

class Image(Density):

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

    def get_gradient_RMSD(self, mol, psim, x=None, q=None, angles=None):
        # NB : coord = x0 + q*A +x
        coord = copy.copy(mol.coords)
        pix, n_pix = src.functions.select_voxels(coord, self.size, self.voxel_size, self.threshold)
        pdiff = psim - self.data

        res = ()
        if x is not None:
            dx = np.zeros(coord.shape)
            coord += x
            res += (dx,)
        if q is not None:
            dq = np.zeros(mol.modes.shape[1])
            coord += np.dot(q, mol.modes)
            res += (dq,)

        for i in range(mol.n_atoms):
            coord_grid = np.repeat(coord[i, :2], n_pix ** 2).reshape(2, n_pix, n_pix)
            mu_grid = (np.mgrid[pix[i, 0]:pix[i, 0] + n_pix,
                  pix[i, 1]:pix[i, 1] + n_pix] - self.size / 2) * self.voxel_size
            tmp = 2 * pdiff[pix[i, 0]:pix[i, 0] + n_pix,
                      pix[i, 1]:pix[i, 1] + n_pix] * np.exp(
                -np.square(np.linalg.norm(coord_grid - mu_grid, axis=0)) / (2 * (self.sigma ** 2)))

            dpsim = -(1 / (self.sigma ** 2)) * (coord_grid - mu_grid) * np.array([tmp, tmp])
            if x is not None:
                dx[i, :2] = dpsim
            if q is not None:
                dq += np.dot(mol.modes[i, :, :2], np.sum(dpsim, axis=(1, 2)))

        return res

    def show(self):
        src.viewers.image_viewer(self)