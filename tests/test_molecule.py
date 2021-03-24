import unittest
import numpy as np
from src.molecule import Molecule, MoleculeStructure, MoleculeForcefieldPrm
from src.constants import *

class TestMoleculeLoad(unittest.TestCase):
    def test_import(self):
        import src.molecule

    def test_load_AK(self):
        ak = Molecule.from_file("tests_data/input/AK/AK.pdb")
        self.assertEqual(ak.n_atoms, 1656)
        self.assertEqual(ak.n_chain, 1)
        self.assertEqual(ak.coarse_grained, False)
        self.assertEqual(len(ak.coords.shape), 2)

    def test_load_P97(self):
        p97 = Molecule.from_file("tests_data/input/P97/5ftm.pdb")
        self.assertEqual(p97.n_atoms, 34200)
        self.assertEqual(p97.n_chain, 6)
        self.assertEqual(p97.coarse_grained, False)
        self.assertEqual(len(p97.coords.shape), 2)

    def test_copy(self):
        ak = Molecule.from_file("tests_data/input/AK/AK.pdb")
        ak2 =Molecule.from_molecule(ak)
        ak2.coords[0,0] = 1
        self.assertTrue(ak.coords[0,0] != ak2.coords[0,0])

class TestMoleculeSelection(unittest.TestCase):
    def test_get_chain(self):
        p97 = Molecule.from_file("tests_data/input/P97/5ftm.pdb")
        chainA = p97.get_chain(0)
        self.assertEqual(len(chainA),5700)

    def test_select_chain(self):
        p97 = Molecule.from_file("tests_data/input/P97/5ftm.pdb")
        cp = Molecule.from_molecule(p97)
        cp.select_chain([0,2,4])
        self.assertEqual(cp.n_atoms, 5700*3)
        self.assertEqual(cp.coords[5700,0], p97.coords[5700*2,0])
        self.assertEqual(cp.coords[5700*2,0], p97.coords[5700*4,0])

    def test_select_atoms(self):
        ak = Molecule.from_file("tests_data/input/AK/AK.pdb")
        cp = Molecule.from_molecule(ak)
        cp.select_atoms(pattern="CA")
        self.assertEqual(cp.n_atoms, 214)
        self.assertEqual(cp.coords.shape[0], 214)
        self.assertEqual(cp.coords[0,0], ak.coords[1,0])

    def test_select_modes(self):
        ak = Molecule.from_file("tests_data/input/AK/AK.pdb")
        fnModes = ["../data/AK/modes/vec." + str(i + 7) for i in range(3)]
        ak.add_modes(fnModes)
        cp = Molecule.from_molecule(ak)
        cp.select_modes([0,2])
        self.assertEqual(cp.modes.shape[1], 2)
        self.assertEqual(cp.modes[0,1,0],ak.modes[0,2,0])

class TestMoleculeChangeGeometry(unittest.TestCase):
    def test_rotate(self):
        ak = Molecule.from_file("tests_data/input/AK/AK.pdb")
        cp = Molecule.from_molecule(ak)
        cp.rotate([2* np.pi,2* np.pi,-2* np.pi])
        self.assertAlmostEqual(cp.coords[0,0], ak.coords[0,0])
        cp.rotate([1,1,1])
        self.assertNotEqual(cp.coords[0,0], ak.coords[0,0])

    def test_center(self):
        ak = Molecule.from_file("tests_data/input/AK/AK.pdb")
        ak.center_structure()
        self.assertAlmostEqual(ak.coords.mean(), 0)
        self.assertAlmostEqual(ak.coords.mean(axis=0)[0], 0)

class TestMoleculeStructure(unittest.TestCase):

    def test_structure_default(self):
        ak = Molecule.from_file("tests_data/input/AK/AK.pdb")
        ak.select_atoms()
        psf = MoleculeStructure.from_default(chain_id=ak.chain_id)
        self.assertEqual(len(psf.bonds) , 213)
        self.assertEqual(len(psf.angles) , 212)
        self.assertEqual(len(psf.dihedrals) , 211)

    def test_structure_file(self):
        psf = MoleculeStructure.from_psf_file(file="tests_data/input/AK/AK.psf")
        self.assertEqual(len(psf.bonds),3365)
        self.assertEqual(len(psf.angles),6123)
        self.assertEqual(len(psf.dihedrals),8921)

class TestMoleculeForceFieldPrm(unittest.TestCase):

    def test_forcefieldprm_default(self):
        ak = Molecule.from_file("tests_data/input/AK/AK.pdb")
        ak.select_atoms()
        psf = MoleculeStructure.from_default(chain_id=ak.chain_id)
        prm = MoleculeForcefieldPrm.from_default(psf)
        self.assertEqual(prm.mass[0], CARBON_MASS)
        self.assertEqual(len(prm.b0) , 213)
        self.assertEqual(len(prm.Theta0) , 212)
        self.assertEqual(len(prm.delta) , 211)

    def test_forcefieldprm_file(self):
        psf = MoleculeStructure.from_psf_file(file="tests_data/input/AK/AK.psf")
        prm = MoleculeForcefieldPrm.from_prm_file(psf, "tests_data/input/par_all36_prot.prm")
        self.assertTrue(prm.mass[4]- CARBON_MASS< 0.1)
        self.assertEqual(len(prm.b0),3365)
        self.assertEqual(len(prm.Theta0),6123)
        self.assertEqual(len(prm.delta),8921)

class TestMoleculeForceField(unittest.TestCase):

    def test_force_field_default(self):
        ak = Molecule.from_file("tests_data/input/AK/AK.pdb")
        ak.select_atoms()
        ak.set_forcefield()
        self.assertEqual(ak.prm.mass[0], CARBON_MASS)
        self.assertEqual(len(ak.prm.b0) , 213)

    def test_force_field_files(self):
        ak = Molecule.from_file("tests_data/input/AK/AK.pdb")
        ak.set_forcefield(psf_file="../data/AK/AK.psf", prm_file="tests_data/input/par_all36_prot.prm")
        self.assertTrue(ak.prm.mass[4]- CARBON_MASS< 0.1)
        self.assertEqual(len(ak.prm.b0) , 3365)








