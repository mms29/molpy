import unittest
import numpy as np
from src.molecule import Molecule, MoleculeForceField
from src.constants import *

class TestMoleculePDB(unittest.TestCase):
    def test_load_AK(self):
        ak = Molecule("tests_data/input/AK/AK.pdb")
        self.assertEqual(ak.n_atoms, 1656)
        self.assertEqual(len(ak.coords.shape), 2)
        self.assertEqual(ak.atomNum.shape[0], 1656)
        self.assertEqual(ak.chainName.shape[0], 1656)

    def test_load_P97(self):
        p97 = Molecule("tests_data/input/P97/5ftm.pdb")
        self.assertEqual(p97.n_atoms, 34200)
        self.assertEqual(len(p97.coords.shape), 2)
        self.assertEqual(p97.atomNum.shape[0], 34200)
        self.assertEqual(p97.chainName.shape[0], 34200)


    def test_copy(self):
        ak = Molecule("tests_data/input/AK/AK.pdb")
        ak2 = ak.copy()
        ak2.coords[0,0] = 1
        self.assertTrue(ak.coords[0,0] != ak2.coords[0,0])

class TestMoleculeChangeGeometry(unittest.TestCase):
    def test_rotate(self):
        ak = Molecule("tests_data/input/AK/AK.pdb")
        cp = ak.copy()
        cp.rotate([2* np.pi,2* np.pi,-2* np.pi])
        self.assertAlmostEqual(cp.coords[0,0], ak.coords[0,0])
        cp.rotate([1,1,1])
        self.assertNotEqual(cp.coords[0,0], ak.coords[0,0])

    def test_center(self):
        ak = Molecule("tests_data/input/AK/AK.pdb")
        ak.center()
        self.assertAlmostEqual(ak.coords.mean(), 0)
        self.assertAlmostEqual(ak.coords.mean(axis=0)[0], 0)

class TestMoleculeForceField(unittest.TestCase):

    def test_forcefield_carbonalpha(self):
        ak = Molecule("tests_data/input/AK/AK.pdb")
        ak.allatoms2carbonalpha()
        ak.set_forcefield()
        self.assertEqual(len(ak.forcefield.bonds) , 213)
        self.assertEqual(len(ak.forcefield.angles) , 212)
        self.assertEqual(len(ak.forcefield.dihedrals) , 211)
        self.assertEqual(ak.forcefield.mass[0], CARBON_MASS)
        self.assertEqual(len(ak.forcefield.b0) , 213)
        self.assertEqual(len(ak.forcefield.Theta0) , 212)
        self.assertEqual(len(ak.forcefield.delta) , 211)
        self.assertAlmostEqual(ak.get_energy(potentials=["bonds", "angles", "dihedrals"])["total"], 1063.1972616987032)

    def test_forcefield_allatoms(self):
        ak = Molecule("tests_data/input/AK/AK_PSF.pdb")
        ak.set_forcefield(psf_file="tests_data/input/AK/AK.psf", prm_file="tests_data/input/par_all36_prot.prm")
        self.assertEqual(len(ak.forcefield.bonds),3365)
        self.assertEqual(len(ak.forcefield.angles),6123)
        self.assertEqual(len(ak.forcefield.dihedrals),8921)
        self.assertTrue(ak.forcefield.mass[4]- CARBON_MASS< 0.1)
        self.assertEqual(len(ak.forcefield.b0),3365)
        self.assertEqual(len(ak.forcefield.Theta0),6123)
        self.assertEqual(len(ak.forcefield.delta),8921)
        self.assertAlmostEqual(ak.get_energy(potentials=["bonds", "angles", "dihedrals"])["total"], 2751.138252615212)

    def test_forcefield_backbone(self):
        ak = Molecule("tests_data/input/AK/AK_PSF.pdb")
        ak.set_forcefield(psf_file="tests_data/input/AK/AK.psf", prm_file="tests_data/input/par_all36_prot.prm")
        ak.allatoms2backbone()
        self.assertEqual(len(ak.forcefield.bonds),1680)
        self.assertEqual(len(ak.forcefield.angles),2264)
        self.assertEqual(len(ak.forcefield.dihedrals),2638)
        self.assertEqual(len(ak.forcefield.b0),1680)
        self.assertEqual(len(ak.forcefield.Theta0),2264)
        self.assertEqual(len(ak.forcefield.delta),2638)
        self.assertAlmostEqual(ak.get_energy(potentials=["bonds", "angles", "dihedrals"])["total"], 2356.9999160968127)








