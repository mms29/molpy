import mkl
mkl.set_num_threads(1)
import unittest

from src.flexible_fitting import *
from src.molecule import Molecule
from src.density import Volume
from src.functions import get_RMSD_coords

class TestFittingAK(unittest.TestCase):

    def test_fitting_AK_local_carbonalpha(self):
        ak = Molecule("tests_data/input/AK/AK.pdb")
        ak.center()
        ak.allatoms2carbonalpha()
        ak.set_forcefield()

        target = Volume.from_file(file="tests_data/input/AK/ak_nma_carbonalpha.mrc", voxel_size=1.5, sigma=2, cutoff=6)
        params = { "initial_biasing_factor": 100 , "local_dt":2*1e-15, "n_iter":50, "n_warmup":40}
        fit = FlexibleFitting(init=ak, target=target, vars=["local"], params=params, n_chain=4, verbose=2)
        fit.HMC()

        ground_truth = Molecule("tests_data/input/AK/ak_nma_carbonalpha.pdb")
        rmsd = get_RMSD_coords(ground_truth.coords, fit.res["mol"].coords)

        self.assertTrue(rmsd < 1 ) # RMSD < 1 Angstrom
        self.assertTrue(fit.res["mol"].get_energy(potentials=["bonds", "angles", "dihedrals"])["total"]<1500)

    def test_fitting_AK_global_carbonalpha(self):
        ak = Molecule("tests_data/input/AK/AK.pdb")
        ak.center()
        ak.set_normalModeVec(np.array(["tests_data/input/AK/modes/vec." + str(i + 7) for i in range(3)]))
        ak.allatoms2carbonalpha()
        ak.set_forcefield()

        target = Volume.from_file(file="tests_data/input/AK/ak_nma_carbonalpha.mrc", voxel_size=1.5, sigma=2, cutoff=6)
        params = {"initial_biasing_factor": 100, "global_dt": 0.05}
        fit = FlexibleFitting(init=ak, target=target, vars=["global"], params=params, n_chain=4, verbose=2)
        fit.HMC()

        ground_truth = Molecule("tests_data/input/AK/ak_nma_carbonalpha.pdb")
        rmsd = get_RMSD_coords(ground_truth.coords, fit.res["mol"].coords)

        self.assertTrue(rmsd < 1 ) # RMSD < 1 Angstrom
        self.assertAlmostEqual(fit.res["global"][0],100 , places=-1)
        self.assertAlmostEqual(fit.res["global"][1],-100 , places=-1)
        self.assertAlmostEqual(fit.res["global"][2],0 , places=-1)

    def test_fitting_AK_rotation_carbonalpha(self):
        ak = Molecule("tests_data/input/AK/AK.pdb")
        ak.center()
        ak.allatoms2carbonalpha()
        ak.set_forcefield()

        target = Volume.from_file(file="tests_data/input/AK/ak_rotation_carbonalpha.mrc", voxel_size=1.5, sigma=2, cutoff=6)
        params = {"initial_biasing_factor": 100, "rotation_dt": 0.0001}
        fit = FlexibleFitting(init=ak, target=target, vars=["rotation"], params=params, n_chain=4, verbose=2)
        fit.HMC()

        ground_truth = Molecule("tests_data/input/AK/ak_rotation_carbonalpha.pdb")
        rmsd = get_RMSD_coords(ground_truth.coords, fit.res["mol"].coords)

        self.assertTrue(rmsd < 1 ) # RMSD < 1 Angstrom

    def test_fitting_AK_shift_carbonalpha(self):
        ak = Molecule("tests_data/input/AK/AK.pdb")
        ak.center()
        ak.allatoms2carbonalpha()
        ak.set_forcefield()

        target = Volume.from_file(file="tests_data/input/AK/ak_shift_carbonalpha.mrc", voxel_size=1.5, sigma=2, cutoff=6)
        params = {"initial_biasing_factor": 100, "shift_dt": 0.001}
        fit = FlexibleFitting(init=ak, target=target, vars=["shift"], params=params, n_chain=4, verbose=2)
        fit.HMC()

        ground_truth = Molecule("tests_data/input/AK/ak_shift_carbonalpha.pdb")
        rmsd = get_RMSD_coords(ground_truth.coords, fit.res["mol"].coords)

        self.assertTrue(rmsd < 1 ) # RMSD < 1 Angstrom

    def test_fitting_AK_all_carbonalpha(self):
        ak = Molecule("tests_data/input/AK/AK.pdb")
        ak.center()
        ak.set_normalModeVec(np.array(["tests_data/input/AK/modes/vec." + str(i + 7) for i in range(3)]))
        ak.allatoms2carbonalpha()
        ak.set_forcefield()

        target = Volume.from_file(file="tests_data/input/AK/ak_all_carbonalpha.mrc", voxel_size=1.5, sigma=2, cutoff=6)
        params = {
            "initial_biasing_factor": 100,
            "local_dt": 2 * 1e-15,
            "global_dt": 0.05,
            "rotation_dt": 0.0001,
            "shift_dt": 0.001,
            "n_iter": 30,
            "n_warmup": 20,
        }
        fit = FlexibleFitting(init=ak, target=target, vars=["local", "global", "rotation", "shift"], params=params, n_chain=4, verbose=2)
        fit.HMC()

        ground_truth = Molecule("tests_data/input/AK/ak_all_carbonalpha.pdb")
        rmsd = get_RMSD_coords(ground_truth.coords, fit.res["mol"].coords)

        self.assertTrue(rmsd < 1 ) # RMSD < 1 Angstrom
