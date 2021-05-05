import unittest
import autograd.numpy as npg
from autograd import elementwise_grad

from src.molecule import Molecule
from src.density import *
import src.forcefield

def get_auto_CC(params, mol, size, sigma, voxel_size, pexp, **kwargs):
    spsim2 = 0.0
    spexp2 = 0.0
    spsimpexp = 0.0
    coord =src.forcefield.forward_model(params=params, mol=mol, **kwargs)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                s_tmp = 0.0
                for a in range(mol.n_atoms):
                    mu = (npg.array([i, j, k]) - size / 2) * voxel_size
                    expnt = npg.exp(-npg.square(npg.linalg.norm(coord[a] - mu, axis=0)) / (2 * (sigma ** 2)))
                    s_tmp += expnt
                spsim2 += s_tmp ** 2
                spexp2 += pexp[i, j, k] ** 2
                spsimpexp += (pexp[i, j, k] * s_tmp)
    return 1 - (spsimpexp / npg.sqrt(spsim2 * spexp2))


def get_auto_LS(params, mol, size, sigma, voxel_size, pexp, **kwargs):
    s = 0.0
    coord =src.forcefield.forward_model(params=params, mol=mol, **kwargs)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                s_tmp = 0.0
                for a in range(mol.n_atoms):
                    mu = (npg.array([i, j, k]) - size / 2) * voxel_size
                    expnt = npg.exp(-npg.square(npg.linalg.norm(coord[a] - mu, axis=0)) / (2 * (sigma ** 2)))
                    s_tmp += expnt
                s += (s_tmp - pexp[i, j, k]) ** 2
    return s

def get_TestDensityGrad_data():
    ak = Molecule("tests_data/input/AK/AK.pdb")
    ak.center()
    ak.set_normalModeVec(np.array(["tests_data/input/AK/modes/vec." + str(i + 7) for i in range(3)]))
    ak.select_atoms(idx=np.arange(10))
    ak_nma = ak.nma_deform([300, -100, 0])
    target = Volume.from_coords(coord=ak_nma.coords, size=16, voxel_size=8.0, sigma=2.0,
                                cutoff=10.0)
    init = Volume.from_coords(coord=ak.coords, size=target.size, voxel_size=target.voxel_size, sigma=target.sigma,
                              cutoff=target.cutoff)
    return ak, init, target

class TestDensityGrad(unittest.TestCase):

    def test_grad_CC_local(self):
        # get test data
        ak, init, target = get_TestDensityGrad_data()
        params = {"local":np.zeros(ak.coords.shape)}

        # Check if CC is == to auto cc
        cc = 1 - get_CC(init.data, target.data)
        cc_auto = get_auto_CC(params = params, mol=ak,size=target.size,
                    sigma=target.sigma, voxel_size=target.voxel_size, pexp=target.data)
        self.assertAlmostEqual(cc, cc_auto)

        # Check if grad CC is == to autograd CC
        F = get_gradient_CC(mol=ak, psim=init.data, pexp=target, params=params)["local"]
        F_auto = elementwise_grad(get_auto_CC, 0)(params, ak, target.size,
                                                  target.sigma, target.voxel_size, target.data)["local"]
        self.assertAlmostEqual(np.linalg.norm(F+F_auto), 0.0)

    def test_grad_CC_global(self):
        # get test data
        ak, init, target = get_TestDensityGrad_data()
        params = {"global":np.zeros(ak.normalModeVec.shape[1])}

        # Check if CC is == to auto cc
        cc = 1 - get_CC(init.data, target.data)
        cc_auto = get_auto_CC(params = params, mol=ak,size=target.size,
                    sigma=target.sigma, voxel_size=target.voxel_size, pexp=target.data, normalModeVec=ak.normalModeVec)
        self.assertAlmostEqual(cc, cc_auto)

        # Check if grad CC is == to autograd CC
        F = get_gradient_CC(mol=ak, psim=init.data, pexp=target, params=params,normalModeVec=ak.normalModeVec)["global"]
        F_auto = elementwise_grad(get_auto_CC, 0)(params, ak, target.size,
                        target.sigma, target.voxel_size, target.data, normalModeVec=ak.normalModeVec)["global"]
        self.assertAlmostEqual(np.linalg.norm(F+F_auto), 0.0)

    def test_grad_CC_rotation(self):
        # get test data
        ak, init, target = get_TestDensityGrad_data()
        params = {"rotation":np.zeros(3)}

        # Check if CC is == to auto cc
        cc = 1 - get_CC(init.data, target.data)
        cc_auto = get_auto_CC(params = params, mol=ak,size=target.size,
                    sigma=target.sigma, voxel_size=target.voxel_size, pexp=target.data)
        self.assertAlmostEqual(cc, cc_auto)

        # Check if grad CC is == to autograd CC
        F = get_gradient_CC(mol=ak, psim=init.data, pexp=target, params=params)["rotation"]
        F_auto = elementwise_grad(get_auto_CC, 0)(params, ak, target.size,
                        target.sigma, target.voxel_size, target.data)["rotation"]
        self.assertAlmostEqual(np.linalg.norm(F+F_auto), 0.0)

    def test_grad_CC_shift(self):
        # get test data
        ak, init, target = get_TestDensityGrad_data()
        params = {"shift":np.zeros(3)}

        # Check if CC is == to auto cc
        cc = 1 - get_CC(init.data, target.data)
        cc_auto = get_auto_CC(params = params, mol=ak,size=target.size,
                    sigma=target.sigma, voxel_size=target.voxel_size, pexp=target.data)
        self.assertAlmostEqual(cc, cc_auto)

        # Check if grad CC is == to autograd CC
        F = get_gradient_CC(mol=ak, psim=init.data, pexp=target, params=params)["shift"]
        F_auto = elementwise_grad(get_auto_CC, 0)(params, ak, target.size,
                        target.sigma, target.voxel_size, target.data)["shift"]
        self.assertAlmostEqual(np.linalg.norm(F+F_auto), 0.0)


    def test_grad_CC_all(self):
        # get test data
        ak, init, target = get_TestDensityGrad_data()
        params = {"local":np.zeros(ak.coords.shape),
                  "global":np.zeros(ak.normalModeVec.shape[1]),
                  "rotation":np.zeros(3),"shift":np.zeros(3)}

        # Check if CC is == to auto cc
        cc = 1 - get_CC(init.data, target.data)
        cc_auto = get_auto_CC(params = params, mol=ak,size=target.size,
                    sigma=target.sigma, voxel_size=target.voxel_size, pexp=target.data, normalModeVec=ak.normalModeVec)
        self.assertAlmostEqual(cc, cc_auto)

        # Check if grad CC is == to autograd CC
        F = get_gradient_CC(mol=ak, psim=init.data, pexp=target, params=params, normalModeVec=ak.normalModeVec)
        F_auto = elementwise_grad(get_auto_CC, 0)(params, ak, target.size,
                        target.sigma, target.voxel_size, target.data, normalModeVec=ak.normalModeVec)
        for i in params:
            self.assertAlmostEqual(np.linalg.norm(F[i]+F_auto[i]), 0.0)

    def test_grad_LS_local(self):
        # get test data
        ak, init, target = get_TestDensityGrad_data()
        params = {"local":np.zeros(ak.coords.shape)}

        # Check if LS is == to auto LS
        ls = get_LS(init.data, target.data)
        ls_auto = get_auto_LS(params = params, mol=ak,size=target.size,
                    sigma=target.sigma, voxel_size=target.voxel_size, pexp=target.data)
        self.assertAlmostEqual(ls, ls_auto)

        # Check if grad LS is == to autograd LS
        F = get_gradient_LS(mol=ak, psim=init.data, pexp=target, params=params)["local"]
        F_auto = elementwise_grad(get_auto_LS, 0)(params, ak, target.size,
                    target.sigma, target.voxel_size, target.data)["local"]
        self.assertAlmostEqual(np.linalg.norm(F+F_auto), 0.0)

    def test_grad_LS_global(self):
        # get test data
        ak, init, target = get_TestDensityGrad_data()
        params = {"global":np.zeros(ak.normalModeVec.shape[1])}

        # Check if LS is == to auto LS
        LS = get_LS(init.data, target.data)
        LS_auto = get_auto_LS(params = params, mol=ak,size=target.size,
                    sigma=target.sigma, voxel_size=target.voxel_size, pexp=target.data, normalModeVec=ak.normalModeVec)
        self.assertAlmostEqual(LS, LS_auto)

        # Check if grad LS is == to autograd LS
        F = get_gradient_LS(mol=ak, psim=init.data, pexp=target, params=params,normalModeVec=ak.normalModeVec)["global"]
        F_auto = elementwise_grad(get_auto_LS, 0)(params, ak, target.size,
                        target.sigma, target.voxel_size, target.data, normalModeVec=ak.normalModeVec)["global"]
        self.assertAlmostEqual(np.linalg.norm(F+F_auto), 0.0)


    def test_grad_LS_rotation(self):
        # get test data
        ak, init, target = get_TestDensityGrad_data()
        params = {"rotation":np.zeros(3)}

        # Check if LS is == to auto LS
        LS = get_LS(init.data, target.data)
        LS_auto = get_auto_LS(params = params, mol=ak,size=target.size,
                    sigma=target.sigma, voxel_size=target.voxel_size, pexp=target.data)
        self.assertAlmostEqual(LS, LS_auto)

        # Check if grad LS is == to autograd LS
        F = get_gradient_LS(mol=ak, psim=init.data, pexp=target, params=params)["rotation"]
        F_auto = elementwise_grad(get_auto_LS, 0)(params, ak, target.size,
                        target.sigma, target.voxel_size, target.data)["rotation"]
        self.assertAlmostEqual(np.linalg.norm(F+F_auto), 0.0)

    def test_grad_LS_shift(self):
        # get test data
        ak, init, target = get_TestDensityGrad_data()
        params = {"shift":np.zeros(3)}

        # Check if LS is == to auto LS
        LS = get_LS(init.data, target.data)
        LS_auto = get_auto_LS(params = params, mol=ak,size=target.size,
                    sigma=target.sigma, voxel_size=target.voxel_size, pexp=target.data)
        self.assertAlmostEqual(LS, LS_auto)

        # Check if grad LS is == to autograd LS
        F = get_gradient_LS(mol=ak, psim=init.data, pexp=target, params=params)["shift"]
        F_auto = elementwise_grad(get_auto_LS, 0)(params, ak, target.size,
                        target.sigma, target.voxel_size, target.data)["shift"]
        self.assertAlmostEqual(np.linalg.norm(F+F_auto), 0.0)

    def test_grad_LS_all(self):
        # get test data
        ak, init, target = get_TestDensityGrad_data()
        params = {"local":np.zeros(ak.coords.shape),
                  "global":np.zeros(ak.normalModeVec.shape[1]),
                  "rotation":np.zeros(3),"shift":np.zeros(3)}

        # Check if LS is == to auto LS
        LS = get_LS(init.data, target.data)
        LS_auto = get_auto_LS(params = params, mol=ak,size=target.size,
                    sigma=target.sigma, voxel_size=target.voxel_size, pexp=target.data, normalModeVec=ak.normalModeVec)
        self.assertAlmostEqual(LS, LS_auto)

        # Check if grad LS is == to autograd LS
        F = get_gradient_LS(mol=ak, psim=init.data, pexp=target, params=params, normalModeVec=ak.normalModeVec)
        F_auto = elementwise_grad(get_auto_LS, 0)(params, ak, target.size,
                        target.sigma, target.voxel_size, target.data, normalModeVec=ak.normalModeVec)
        for i in params:
            self.assertAlmostEqual(np.linalg.norm(F[i]+F_auto[i]), 0.0)
