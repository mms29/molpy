import unittest
import numpy as np
from src.io import *
from src.molecule import Molecule


class TestIO(unittest.TestCase):

    def test_read_modes(self):
        N = 3
        fnModes = ["tests_data/input/AK/modes/vec." + str(i + 7) for i in range(N)]
        modes = read_modes(fnModes)
        self.assertEqual(modes.shape[0],1656)
        self.assertEqual(modes.shape[1],N)
        self.assertEqual(modes.shape[2],3)

    def test_read_pdb(self):
        f="tests_data/input/P97/5ftm.pdb"
        data= read_pdb(f)
        self.assertEqual(data["coords"].shape[0],34200)
        self.assertEqual(data["atomNum"].shape[0],34200)

    def test_save_pdb(self):
        f = "tests_data/input/P97/5ftm.pdb"
        data = read_pdb(f)
        save_pdb(data, file="tests_data/output/mol.pdb")
        data2 = read_pdb("tests_data/output/mol.pdb")
        self.assertEqual(data2["coords"].shape[0], data["coords"].shape[0])

    def test_read_mrc(self):
        data, voxel_size=  read_mrc("tests_data/input/P97/emd_3299.mrc")
        self.assertAlmostEqual(voxel_size,1.4576250314712524)
        self.assertAlmostEqual(data.shape[0],128)
        self.assertAlmostEqual(data.shape[1],128)
        self.assertAlmostEqual(data.shape[2],128)

    def test_save_mrc(self):
        data, voxel_size=  read_mrc("tests_data/input/P97/emd_3299.mrc")
        save_mrc(data=data, file="tests_data/output/volume.mrc", voxel_size=voxel_size)
        data2, voxel_size2=  read_mrc("tests_data/output/volume.mrc")
        self.assertEqual(data.shape[0], data2.shape[0])
        self.assertEqual(voxel_size, voxel_size2)

    def test_create_psf(self):
        create_psf(pdb_file="tests_data/input/P97/5ftm.pdb", prefix="tests_data/output/psf",
                   topology_file="tests_data/input/top_all36_prot.rtf")
        mol = Molecule("tests_data/output/psf_PSF.pdb")
        self.assertEqual(mol.n_atoms, 68862)
        dic= read_psf("tests_data/output/psf.psf")
        self.assertTrue("bonds" in dic)

    def test_read_psf(self):
        pass
        #TODO

    def test_read_prm(self):
        pass
        #TODO

