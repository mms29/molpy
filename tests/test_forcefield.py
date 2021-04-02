import unittest
import autograd.numpy as npg
from autograd import elementwise_grad


from src.forcefield import *
from src.molecule import Molecule
from src.density import Volume
from src.constants import *

class TestEnergy(unittest.TestCase):
    def test_energy_bonds(self):
        ak = Molecule("tests_data/input/AK/AK_PSF.pdb")
        ak.set_forcefield(psf_file="tests_data/input/AK/AK.psf",prm_file="tests_data/input/par_all36_prot.prm")
        U = get_energy_bonds(coord = ak.coords, forcefield=ak.forcefield)
        self.assertAlmostEqual(U,290.7638265337042)

    def test_energy_angles(self):
        ak = Molecule("tests_data/input/AK/AK_PSF.pdb")
        ak.set_forcefield(psf_file="tests_data/input/AK/AK.psf",prm_file="tests_data/input/par_all36_prot.prm")
        U = get_energy_angles(coord = ak.coords,forcefield=ak.forcefield)
        self.assertAlmostEqual(U,519.3379388536089)

    def test_energy_dihedrals(self):
        ak = Molecule("tests_data/input/AK/AK_PSF.pdb")
        ak.set_forcefield(psf_file="tests_data/input/AK/AK.psf",prm_file="tests_data/input/par_all36_prot.prm")
        U = get_energy_dihedrals(coord = ak.coords, forcefield=ak.forcefield)
        self.assertAlmostEqual(U,1941.0364872278988)

class TestGradientPotential(unittest.TestCase):
    def test_autograd(self):
        ak = Molecule("tests_data/input/AK/AK.pdb")
        fnModes = ["../data/AK/modes/vec." + str(i + 7) for i in range(3)]
        ak.set_normalModeVec(fnModes)
        ak.allatoms2carbonalpha()
        ak.set_forcefield()
        grad = get_autograd(params = {
            FIT_VAR_LOCAL : np.zeros(ak.coords.shape),
            FIT_VAR_GLOBAL : np.zeros(ak.normalModeVec.shape[1]),
            FIT_VAR_ROTATION : np.zeros(3),
            FIT_VAR_SHIFT : np.zeros(3),
        }, mol=ak, potentials=["bonds", "angles", "dihedrals"])
        self.assertAlmostEqual(grad[FIT_VAR_LOCAL][0,0],-0.8017397359949898)
        self.assertAlmostEqual(grad[FIT_VAR_GLOBAL][0],-0.12597578)
        self.assertAlmostEqual(grad[FIT_VAR_ROTATION][0],0)
        self.assertAlmostEqual(grad[FIT_VAR_SHIFT][0],0)

class TestGradientBiased(unittest.TestCase):
    def test_implement_vs_autograd(self):
        #TODO
        pass


