import unittest
import autograd.numpy as npg
from autograd import elementwise_grad

from src.molecule import Molecule
from src.density import *

class TestDensityGrad(unittest.TestCase):

    def test_grad_CC(self):
        ak = Molecule("tests_data/input/AK/AK.pdb")
        ak.center()
        ak.select_atoms(idx=np.arange(1))

        target = Volume.from_file(file="tests_data/input/AK/ak_all_carbonalpha.mrc", voxel_size=1.5, sigma=2, cutoff=10)
        init = Volume.from_coords(coord=ak.coords, size=target.size, voxel_size=target.voxel_size, sigma=target.sigma,
                                  cutoff=target.cutoff)

        def get_auto_CC(coord, size, sigma, voxel_size, pexp):
            spsim2 = 0.0
            spexp2 = 0.0
            spsimpexp = 0.0
            for a in range(coord.shape[0]):
                for i in range(size):
                    print(i)
                    for j in range(size):
                        for k in range(size):
                            mu = (npg.array([i,j,k])- size / 2) * voxel_size
                            expnt= npg.exp(-npg.square(npg.linalg.norm(coord[a] - mu, axis=0)) / (2 * (sigma ** 2)))
                            spsim2 += expnt**2
                            spexp2 += pexp[i,j,k]**2
                            spsimpexp += pexp[i,j,k]* expnt
            return spsimpexp / npg.sqrt(spsim2* spexp2)

        cc = get_CC(init.data, target.data)
        cc_auto = get_auto_CC(ak.coords,target.size, target.sigma, target.voxel_size, target.data)
        print(cc)
        print(cc_auto)

        F = get_gradient_CC(mol=ak, psim=init.data, pexp=target, params={"local":np.zeros(ak.coords.shape)})["local"]
        F_auto = elementwise_grad(get_auto_CC, 0)(ak.coords,target.size, target.sigma, target.voxel_size, target.data)

        print(F)
        print(F_auto)
        print(F-F_auto)

    def test_grad_LS(self):
        ak = Molecule("tests_data/input/AK/AK.pdb")
        ak.center()
        ak.set_normalModeVec(np.array(["tests_data/input/AK/modes/vec." + str(i + 7) for i in range(3)]))
        ak.select_atoms(idx=np.arange(3))
        ak_nma = ak.nma_deform([300,-100,0])

        target = Volume.from_coords(coord=ak_nma.coords, size=16, voxel_size=8.0, sigma=2.0,
                                  cutoff=10.0)
        init = Volume.from_coords(coord=ak.coords, size=target.size, voxel_size=target.voxel_size, sigma=target.sigma,
                                  cutoff=target.cutoff)

        def get_auto_LS(coord, size, sigma, voxel_size, pexp):
            s = 0.0
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        s_tmp=0.0
                        for a in range(coord.shape[0]):
                            mu = (npg.array([i,j,k])- size / 2) * voxel_size
                            expnt= npg.exp(-npg.square(npg.linalg.norm(coord[a] - mu, axis=0)) / (2 * (sigma ** 2)))
                            s_tmp += expnt
                        s += (s_tmp - pexp[i,j,k])**2
            return s

        ls = get_LS(init.data, target.data)
        ls_auto = get_auto_LS(ak.coords,target.size, target.sigma, target.voxel_size, target.data)
        self.assertAlmostEqual(ls, ls_auto,places=3)

        F = get_gradient_CC(mol=ak, psim=init.data, pexp=target, params={"local":np.zeros(ak.coords.shape)})["local"]
        F_auto = elementwise_grad(get_auto_LS, 0)(ak.coords,target.size, target.sigma, target.voxel_size, target.data)

        s = 0.0
        for i in range(ak.n_atoms):
            s+=np.dot(F[i],F_auto[i]) #/(np.linalg.norm(F[i])*np.linalg.norm(F_auto[i]))
            print(np.linalg.norm(F[i]) / np.linalg.norm(F_auto[i]))
        self.assertAlmostEqual(s/ak.n_atoms, 1.0,places=3)

        self.assertAlmostEqual(np.linalg.norm(F-F_auto), 0.0,places=3)

