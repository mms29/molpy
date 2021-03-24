import time
import multiprocessing
import multiprocessing.pool
from multiprocessing import Pool
import copy
import pickle
import numpy as np

import src.constants
from src.constants import FIT_VAR_LOCAL,FIT_VAR_GLOBAL, FIT_VAR_ROTATION, FIT_VAR_SHIFT, \
                                KCAL_TO_JOULE, AVOGADRO_CONST, ATOMIC_MASS_UNIT,K_BOLTZMANN
import src.density
import src.forcefield
import src.functions
import src.molecule
from src.viewers import fit_viewer, chimera_fit_viewer

class FlexibleFitting:
    """
    Perform flexible fitting between initial atomic structure and target Density using HMC
    """

    def __init__(self, init, target, vars, n_chain, params, verbose=0):
        """
        Constructor
        :param init: inititial atomic structure Molecule
        :param target: target Density
        :param vars: list of fitted variables
        :param n_chain: number of chain
        :param params: fit parameters
        :param verbose: verbose level
        """
        self.init = init
        self.target = target
        self.n_chain = n_chain
        self.verbose = verbose
        self.vars = vars
        self._set_init_fit_params(params)

    def HMC(self):
        """
        Run HMC fitting with the specified number of chain in parallel
        :return: FlexibleFitting
        """
        with Pool(self.n_chain) as p:
            # launch n_chain times HMC_chain()
            fittings = p.map(FlexibleFitting.HMC_chain, [self for i in range(self.n_chain)])
            p.close()
            p.join()

        # Regroup the chains results
        self.res = {"mol": src.molecule.Molecule.from_molecule(self.init)}
        self.res["mol"].coords = np.mean([fittings[i].res["mol"].coords for i in range(self.n_chain)], axis=0)
        for v in self.vars:
            self.res[v] = np.mean([fittings[i].res[v] for i in range(self.n_chain)], axis=0)
        self.fit = [fittings[i].fit for i in range(self.n_chain)]
        return self

    def HMC_chain(self):
        """
        Run one HMC chain fitting
        :return: FlexibleFitting
        """

        # set the random seed of numpy for parallel computation
        np.random.seed()
        t = time.time()

        # initialize fit variables
        self.fit= {"coord":[copy.deepcopy(self.init.coords)]}
        for i in self.vars:
            self._set(i ,[self.params[i+"_init"]])

        # HMC Loop
        for i in range(self.params["n_iter"]):
            if self.verbose > 0 : print("HMC ITER = " + str(i))
            self.HMC_step()

        # Generate results
        self.res = {"mol" : src.molecule.Molecule.from_molecule(self.init)}
        self.res["mol"].coords = np.mean(np.array(self.fit["coord"][self.params["n_warmup"] + 1:]), axis=0)
        for i in self.vars:
            self.res[i] = np.mean(np.array(self.fit[i][self.params["n_warmup"]+1:]), axis=0)

        # End
        if self.verbose >0 or self.verbose==-1:
            print("############### HMC FINISHED ##########################")
            print("### Total execution time : "+str(time.time()-t)+" s")
            print("### Initial CC value : "+str(self.fit["CC"][0]))
            print("### Mean CC value : "+str(np.mean(self.fit["CC"][self.params["n_warmup"]:])))
            print("#######################################################")

        return self

    def _set_init_fit_params(self, params):
        """
        Set initial parameters of the fitting
        :param params: dict of parameters
        """
        default_params = copy.deepcopy(src.constants.DEFAULT_FIT_PARAMS)
        if FIT_VAR_LOCAL in self.vars:
            default_params[FIT_VAR_LOCAL+"_init"] = np.zeros(self.init.coords.shape)
        if FIT_VAR_GLOBAL in self.vars:
            default_params[FIT_VAR_GLOBAL+"_init"] = np.zeros(self.init.modes.shape[1])
        if FIT_VAR_ROTATION in self.vars:
            default_params[FIT_VAR_ROTATION+"_init"] = np.zeros(3)
        if FIT_VAR_SHIFT in self.vars:
            default_params[FIT_VAR_SHIFT+"_init"] = np.zeros(3)

        default_params.update(params)
        if (FIT_VAR_LOCAL in self.vars) and (not FIT_VAR_LOCAL + "_sigma" in params):
            default_params[FIT_VAR_LOCAL + "_sigma"] = (np.ones((3, self.init.n_atoms)) *
                                                    np.sqrt((K_BOLTZMANN * default_params["temperature"]) /
                                                            (self.init.prm.mass * ATOMIC_MASS_UNIT)) * 1e10).T
        self.params = default_params

    def _get(self, key):
        if isinstance(self.fit[key], list):
            return self.fit[key][-1]
        else:
            return self.fit[key]

    def _add(self, key, value):
        if key in self.fit:
            self.fit[key].append(value)
        else:
            self.fit[key] = [value]

    def _set(self, key, value):
        self.fit[key] = value

    def _remove(self, key):
        del self.fit[key]

    def _set_energy(self, psim):
        """
        Compute the total energy from the simulated Density
        :param psim: the simulated Density
        """
        U = 0
        t = time.time()

        # Biased Potential
        U_biased = src.functions.get_RMSD(psim=psim, pexp=self.target.data) * self.params["biasing_factor"]
        self._add("U_biased", U_biased)
        U+= U_biased

        # Energy Potential
        U_potential = src.forcefield.get_energy(coord=self._get("coord_t"), molstr = self.init.psf,
                                 molprm= self.init.prm, verbose=False)* self.params["potential_factor"]
        self._add("U_potential", U_potential)
        U += U_potential

        # Additional Priors on parameters
        for i in self.vars:
            if i+"_factor" in self.params:
                U_var = np.sum(np.square(self._get(i+"_t"))) * self.params[i+"_factor"]
                self._add("U_"+i, U_var)
                U += U_var

        # Total energy
        self._add("U", U)
        if self.verbose>=3: print("Energy="+str(time.time()-t))

    def _set_gradient(self, psim):
        """
        Compute the gradient of the total energy from the simulated Density
        :param psim: simulated Density
        """
        t = time.time()
        vals={}
        for i in self.vars:
            vals[i] = self._get(i+"_t")

        dU_biased = self.target.get_gradient_RMSD(self.init, psim, vals)
        dU_potential = src.forcefield.get_autograd(params=vals, mol = self.init)

        for i in self.vars:
            F = -((self.params["biasing_factor"] * dU_biased[i]) + (self.params["potential_factor"] *  dU_potential[i]))
            if i == FIT_VAR_LOCAL:
                F = (F.T * (1 / (self.init.prm.mass * ATOMIC_MASS_UNIT))).T  # Force -> acceleration
                F *= (KCAL_TO_JOULE / AVOGADRO_CONST)  # kcal/mol -> Joule
                F *= 1e20  # kg * m2 * s-2 -> kg * A2 * s-2
            if i+"_factor" in self.params:
                F+=  - 2* self._get(i+"_t") * self.params[i+"_factor"]

            if not i+"_F" in self.fit: self._set(i+"_F", F)
            else: self._set(i+"_Ft", F)
        if self.verbose >= 3: print("Gradient=" + str(time.time() - t))

    def _set_kinetic(self):
        """
        Compute the Kinetic energy
        """
        K=0
        for i in self.vars:
            if i == FIT_VAR_LOCAL:
                K+= 1 / 2 * np.sum((self.init.prm.mass*ATOMIC_MASS_UNIT)*np.square(self._get(i+"_v")).T)
            else:
                K =  1 / 2 * np.sum(self.params[i+"_sigma"]*np.square(self._get(i+"_v")))

        # kg * A2 * s-2 -> kcal * mol-1
        K *= 1e-20*(AVOGADRO_CONST /KCAL_TO_JOULE)
        self._add("K", K)

    def _set_instant_temp(self):
        """
        Compute instant temperature
        """
        T = 2 * self._get("K")*(KCAL_TO_JOULE/AVOGADRO_CONST ) / (K_BOLTZMANN * 3 * self.init.n_atoms)
        self._add("T", T)

    def _set_criterion(self):
        """
        Compute NUTS criterion (optional)
        """
        C = 0
        if self.params["criterion"]:
            for i in self.vars:
                C += np.dot((self._get(i+"_t").flatten() - self._get(i).flatten()), self._get(i+"_v").flatten())
            self._add("C",C)
        else:
            self._add("C", 0)

    def _set_density(self):
        """
        Compute the density (Image or Volume)
        :return: Density object (Image or Volume)
        """
        t = time.time()
        if isinstance(self.target, src.density.Volume):
            density = src.density.Volume.from_coords(coord=self._get("coord_t"), size=self.target.size,
                                      voxel_size=self.target.voxel_size,
                                      sigma=self.target.sigma, threshold=self.target.threshold).data
        else:
            density = src.density.Image.from_coords(coord=self._get("coord_t"), size=self.target.size,
                                      voxel_size=self.target.voxel_size,
                                      sigma=self.target.sigma, threshold=self.target.threshold).data
        if self.verbose >= 3: print("Density=" + str(time.time() - t))
        return density

    def _get_hamiltonian(self):
        return self._get("U") +  self._get("K")

    def _initialize(self):
        """
        Initialize all variables positions and velocities
        """
        for i in self.vars:
            self._set(i+"_t", self._get(i))
            self._set(i+"_v", np.random.normal(0, self.params[i+"_sigma"], self._get(i).shape))

    def _update_positions(self):
        """
        Update all variables positions
        """
        for i in self.vars:
            self._set(i+"_t", self._update_pstep(self._get(i + "_t"), self._get(i + "_v"), self.params[i + "_dt"],
                                               self._get(i + "_F")))

    def _update_pstep(self, x, v, dx, F):
        return x+ dx*v + dx**2 *(F/2)

    def _update_velocities(self):
        """
        Update all variables velocities
        """
        for i in self.vars:
            self._set(i+"_v", self._update_vstep(self._get(i+"_v") , self.params[i+"_dt"] , self._get(i+"_F") ,self._get(i+"_Ft")))
            self._set(i+"_F" ,self._get(i+"_Ft"))

    def _update_vstep(self, v, dx, F, Ft):
        return v + dx*((F+Ft)/2)

    def _forward_model(self):
        """
        Compute the forward model
        """
        coord = copy.deepcopy(self.init.coords)
        if FIT_VAR_LOCAL in self.vars:
            coord += self._get(FIT_VAR_LOCAL+"_t")
        if FIT_VAR_GLOBAL in self.vars:
            coord += np.dot(self._get(FIT_VAR_GLOBAL+"_t"), self.init.modes)
        if FIT_VAR_ROTATION in self.vars:
            coord = np.dot( src.functions.generate_euler_matrix(self._get(FIT_VAR_ROTATION+"_t")),  coord.T).T
        if FIT_VAR_SHIFT  in self.vars:
            coord += self._get(FIT_VAR_SHIFT+"_t")
        self._add("coord_t", coord)

    def _acceptation(self,H, H_init):
        """
        Perform Metropolis Acceptation step
        :param H: Current Hamiltonian
        :param H_init: Initial Hamiltonian
        """
        # Set acceptance value
        self._add("accept",  np.min([1, H_init/H]) )

        # Update variables
        if self._get("accept") > np.random.rand() :
            suffix = "_t"
            if self.verbose > 1 : print("ACCEPTED " + str(self._get("accept")))
        else:
            suffix = ""
            if self.verbose > 1 : print("REJECTED " + str(self._get("accept")))
        for i in self.vars:
            self._add(i, self._get(i+suffix))
        self._add("coord", self._get("coord"+suffix))

        # clean forces
        for i in self.vars:
            self._remove(i + "_F")


    def _print_step(self):
        """
        Print step information
        """
        s=["L", "U_biased", "U_potential", "CC", "Time", "K", "T"]
        for i in self.vars :
            if i+"_factor" in self.params:
                s.append("U_"+i)
        printed_string=""
        for i in s:
            printed_string += i + "=" + str(self._get(i)) + " ; "
        print(printed_string)

    def HMC_step(self):
        """
        Run HMC iteration
        """
    # Initial coordinates
        self._initialize()
    # Compute Forward model
        self._forward_model()
    # initial density
        psim = self._set_density()
    # Initial Potential Energy
        self._set_energy(psim)
    # Initial gradient
        self._set_gradient(psim)
    # Initial Kinetic Energy
        self._set_kinetic()
    # Initial Hamiltonian
        H_init = self._get_hamiltonian()
    # Init vars
        self._add("C",0)
        self._add("L", 0)
    # MD loop
        while (self._get("C") >= 0 and self._get("L")< self.params["n_step"]):
            tt = time.time()
        # Coordinate update
            self._update_positions()
        # Compute Forward model
            self._forward_model()
        # Density update
            psim = self._set_density()
        # CC update
            self._add("CC", src.functions.cross_correlation(psim, self.target.data))
        # Potential energy update
            self._set_energy(psim)
        # Gradient Update
            self._set_gradient(psim)
        # velocities update
            self._update_velocities()
        # Kinetic update
            self._set_kinetic()
        # Temperature update
            self._set_instant_temp()
        # criterion update
            self._set_criterion()
            self.fit["L"][-1] +=1
        # Prints
            self._add("Time", time.time() -tt)
            if self.verbose > 1:
                self._print_step()
    # Hamiltonian update
        H = self._get_hamiltonian()
    # Metropolis acceptation
        self._acceptation(H, H_init)

    def show(self,save=None):
        """
        Show fitting statistics
        """
        fit_viewer(self,save=save)

    def show_3D(self):
        """
        Show fitting results in 3D
        """
        chimera_fit_viewer(self.res["mol"], self.target)

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(file=f, obj=self)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            fit = pickle.load(file=f)
            return fit


def multiple_fitting(models, n_chain, n_proc, save_dir=None):
    class NoDaemonProcess(multiprocessing.Process):
        @property
        def daemon(self):
            return False

        @daemon.setter
        def daemon(self, value):
            pass

    class NoDaemonContext(type(multiprocessing.get_context())):
        Process = NoDaemonProcess

    class NestablePool(multiprocessing.pool.Pool):
        def __init__(self, *args, **kwargs):
            kwargs['context'] = NoDaemonContext()
            super(NestablePool, self).__init__(*args, **kwargs)

    models = np.array(models)

    N = len(models)
    n_loop = (N * n_chain) // n_proc
    n_last_process = ((N * n_chain) % n_proc)//n_chain
    n_process = n_proc//n_chain
    process = [np.arange(i*n_process, (i+1)*n_process) for i in range(n_loop)]
    process.append(np.arange(n_loop*n_process, n_loop*n_process + n_last_process))
    fits=[]
    print("Number of loops : "+str(n_loop))
    for i in process:
        t = time.time()
        print("\t fitting models # "+str(i))
        try :
            with NestablePool(n_process) as p:
                fits += p.map(FlexibleFitting.HMC, models[i])
                p.close()
                p.join()
        except RuntimeError as rte:
            print(str(rte.args))
        print("\t\t done : "+str(time.time()-t))
        if save_dir is not None:
            for n in i:
                fits[n].show(save=save_dir+"fit_#"+str(n))
    return fits






