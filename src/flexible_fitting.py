import time
import multiprocessing
import multiprocessing.pool
from multiprocessing import Pool
import copy
import pickle
import numpy as np

import src.constants
from src.constants import FIT_VAR_LOCAL,FIT_VAR_GLOBAL, FIT_VAR_ROTATION, FIT_VAR_SHIFT, \
                                KCAL_TO_JOULE, AVOGADRO_CONST, ATOMIC_MASS_UNIT,K_BOLTZMANN, ANGSTROM_TO_METER
import src.density
import src.forcefield
import src.functions
import src.molecule
from src.viewers import fit_viewer, chimera_fit_viewer

class FlexibleFitting:
    """
    Perform flexible fitting between initial atomic structure and target Density using HMC
    """

    def __init__(self, init, target, vars, n_chain, params, verbose=0, prefix=None):
        """
        Constructor
        :param init: inititial atomic structure Molecule
        :param target: target Density
        :param vars: list of fitted variables
        :param n_chain: number of chain
        :param params: fit parameters
        :param verbose: verbose level
        :param prefix: prefix path of outputs
        """
        self.init = init
        self.target = target
        self.n_chain = n_chain
        self.verbose = verbose
        self.vars = vars
        self._set_init_fit_params(params)
        self.prefix = prefix

    def HMC(self):
        """
        Run HMC fitting with the specified number of chain in parallel
        :return: FlexibleFitting
        """
        with Pool(self.n_chain) as p:
            # launch n_chain times HMC_chain()
            fits = p.starmap(FlexibleFitting.HMC_chain, [(self,i) for i in range(self.n_chain)])
            p.close()
            p.join()

        # Regroup the chains results
        self.res = {"mol": self.init.copy()}
        self.res["mol"].coords = np.mean([i.res["mol"].coords for i in fits], axis=0)
        for v in self.vars:
            self.res[v] = np.mean([i.res[v] for i in fits], axis=0)
        self.fit = [i.fit for i in fits]
        if self.prefix is not None:
            self.res["mol"].save_pdb(file=self.prefix+"_output.pdb")
            self.save(file=self.prefix+"_output.pkl")

        return self

    def HMC_chain(self, chain_id=0):
        """
        Run one HMC chain fitting
        :param chain_id: chain index
        :return: FlexibleFitting
        """

        # set the random seed of numpy for parallel computation
        np.random.seed()

        t = time.time()
        self.chain_id = chain_id

        # initialize fit variables
        self.fit= {"coord":[copy.deepcopy(self.init.coords)]}
        for i in self.vars:
            self._set(i ,[self.params[i+"_init"]])

        # HMC Loop
        try :
            for i in range(self.params["n_iter"]):
                if self.verbose > 0 :
                    s = "HMC iter : " + str(i) + " | Chain id : " + str(chain_id)
                    if self.prefix is not None:
                        s = "["+self.prefix +"] "+s
                    print(s)
                self.HMC_step()

        except RuntimeError as rte:
            s = "Failed to run HMC chain : " + str(rte.args[0])
            if self.prefix is not None:
                s = "["+self.prefix +"] "+s
            print(s)
            self.res = {"mol" : self.init.copy()}
            for i in self.vars:
                self.res[i] = self._get(i)

        else:
            # Generate results
            self.res = {"mol": self.init.copy()}
            self.res["mol"].coords = np.mean(np.array(self.fit["coord"][self.params["n_warmup"] + 1:]), axis=0)
            for i in self.vars:
                self.res[i] = np.mean(np.array(self.fit[i][self.params["n_warmup"]+1:]), axis=0)

            # End
            if self.verbose >0 :
                print("############### HMC FINISHED ##########################")
                print("### Total execution time : "+str(time.time()-t)+" s")
                print("### Initial CC value : "+str(self.fit["CC"][0]))
                print("### Mean CC value : "+str(np.mean(self.fit["CC"][self.params["n_warmup"]:])))
                print("#######################################################")

            # Cleaning
            # for i in range(len(self.fit["coord"])):
            #     if i%10 != 0:
            #         del (self.fit["coord"])[i]
            del self.fit["coord_t"]
            del self.fit["psim"]
            del self.fit["expnt"]
            for i in self.vars:
                del self.fit[i]
                del self.fit[i+"_v"]
                del self.fit[i+"_t"]
                del self.fit[i+"_Ft"]

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
            default_params[FIT_VAR_GLOBAL+"_init"] = np.zeros(self.init.normalModeVec.shape[1])
        if FIT_VAR_ROTATION in self.vars:
            default_params[FIT_VAR_ROTATION+"_init"] = np.zeros(3)
        if FIT_VAR_SHIFT in self.vars:
            default_params[FIT_VAR_SHIFT+"_init"] = np.zeros(3)

        default_params.update(params)
        if (FIT_VAR_LOCAL in self.vars) and (not FIT_VAR_LOCAL + "_sigma" in params):
            default_params[FIT_VAR_LOCAL + "_sigma"] = (np.ones((3, self.init.n_atoms)) *
                                                    np.sqrt((K_BOLTZMANN * default_params["temperature"]) /
                                                            (self.init.forcefield.mass * ATOMIC_MASS_UNIT)) * ANGSTROM_TO_METER**-1).T
        if "initial_biasing_factor" in default_params:
            default_params["biasing_factor"] = self._set_factor(default_params["initial_biasing_factor"], potentials=default_params["potentials"])

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

    def _set_energy(self):
        """
        Compute the total energy from the simulated Density
        """
        U = 0
        t = time.time()

        # Biased Potential
        U_biased = src.functions.get_RMSD(psim=self._get("psim").data, pexp=self.target.data) * self.params["biasing_factor"]
        self._add("U_biased", U_biased)
        U+= U_biased

        # Energy Potential
        if self.verbose>3: verbose =True
        else: verbose = False
        U_potential = src.forcefield.get_energy(coords=self._get("coord_t"), forcefield=self.init.forcefield,
                potentials=self.params["potentials"], pairlist=self._get("pairlist"), verbose=verbose)
        for i in U_potential:
            if i == "total":
                self._add("U_potential", U_potential["total"]* self.params["potential_factor"])
            else:
                self._add("U_"+i, U_potential[i])
        U += self._get("U_potential")

        # Additional Priors on parameters
        for i in self.vars:
            if i+"_factor" in self.params:
                U_var = np.sum(np.square(self._get(i+"_t"))) * self.params[i+"_factor"]
                self._add("U_"+i, U_var)
                U += U_var

        # Total energy
        self._add("U", U)
        if self.verbose>=3: print("Energy="+str(time.time()-t))

    def _set_gradient(self):
        """
        Compute the gradient of the total energy from the simulated Density
        """
        t = time.time()
        vals={}
        for i in self.vars:
            vals[i] = self._get(i+"_t")

        dU_biased = src.forcefield.get_gradient_RMSD(mol=self.init, psim=self._get("psim"), pexp =self.target, params=vals,
                                                     expnt = self._get("expnt"), normalModeVec=self.init.normalModeVec)
        dU_potential = src.forcefield.get_autograd(params=vals, mol = self.init, normalModeVec=self.init.normalModeVec,
                                                   potentials=self.params["potentials"], pairlist=self._get("pairlist"))

        for i in self.vars:
            F = -((self.params["biasing_factor"] * dU_biased[i]) + (self.params["potential_factor"] *  dU_potential[i]))
            if i == FIT_VAR_LOCAL:
                F = (F.T * (1 / (self.init.forcefield.mass * ATOMIC_MASS_UNIT))).T  # Force -> acceleration
                F *= (KCAL_TO_JOULE / AVOGADRO_CONST)  # kcal/mol -> Joule
                F *= ANGSTROM_TO_METER**-2  # kg * m2 * s-2 -> kg * A2 * s-2
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
                K +=  1 / 2 * np.sum((self.init.forcefield.mass*ATOMIC_MASS_UNIT)*np.square(self._get(i+"_v")).T)

            else:
                K +=  1 / 2 * np.sum(np.square(self._get(i+"_v"))/self.params[i+"_sigma"])

        K *= ANGSTROM_TO_METER**2 *(AVOGADRO_CONST /KCAL_TO_JOULE)# kg * A2 * s-2 -> kcal * mol-1
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
        # if isinstance(self.target, src.density.Volume):
        #     else:
        #     psim = src.density.Image.from_coords(coord=self._get("coord_t"), size=self.target.size,
        #                                          voxel_size=self.target.voxel_size,
        #                                          sigma=self.target.sigma, cutoff=self.target.cutoff)
        psim, expnt = src.functions.pdb2vol(coord=self._get("coord_t"), size=self.target.size,
                                  voxel_size=self.target.voxel_size,
                                  sigma=self.target.sigma, cutoff=self.target.cutoff)

        if self.verbose >= 3: print("Density=" + str(time.time() - t))
        self._set("psim",psim)
        self._set("expnt",expnt)

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
            coord += np.dot(self._get(FIT_VAR_GLOBAL+"_t"), self.init.normalModeVec)
        if FIT_VAR_ROTATION in self.vars:
            coord = np.dot(src.functions.generate_euler_matrix(self._get(FIT_VAR_ROTATION+"_t")),  coord.T).T
        if FIT_VAR_SHIFT  in self.vars:
            coord += self._get(FIT_VAR_SHIFT+"_t")
        self._set("coord_t", coord)

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

    def _set_pairlist(self):
        """
        Generate Non-bonded pairlist based on cutoff parameters
        """
        if "vdw" in self.params["potentials"] or "elec" in self.params["potentials"]:
            if not "coord_pl" in self.fit:
                self._set("coord_pl", self._get("coord_t"))
            dx_max =np.max(np.linalg.norm(self._get("coord_pl")-self._get("coord_t"), axis=1))/2
            if (dx_max > (self.params["cutoffpl"] - self.params["cutoffnb"])) or (not "pairlist" in self.fit):
                if self.verbose >1 : print("Computing pairlist ...")
                t=time.time()
                self._set("pairlist", src.forcefield.get_pairlist(self._get("coord_t"),
                        excluded_pairs= self.init.forcefield.excluded_pairs,cutoff=self.params["cutoffpl"]))
                self._set("coord_pl",self._get("coord_t"))
                if self.verbose > 1: print("Done "+str(time.time()-t)+" s")
        else:
            self._set("pairlist",None)


    def HMC_step(self):
        """
        Run HMC iteration
        """
    # Initial coordinates
        self._initialize()
    # Compute Forward model
        self._forward_model()
    # initial density
        self._set_density()
        self._set_density()
    # Check pairlist
        self._set_pairlist()
    # Initial Potential Energy
        self._set_energy()
    # Initial gradient
        self._set_gradient()
    # Initial Kinetic Energy
        self._set_kinetic()
    # Temperature update
        self._set_instant_temp()
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
            self._set_density()
        # CC update
            self._add("CC", src.functions.cross_correlation(self._get("psim").data, self.target.data))
            if "target_coords" in self.params:
                self._add("RMSD", src.functions.get_RMSD_coords(self._get("coord_t"), self.params["target_coords"]))
        # Check pairlist
        #     self._set_pairlist()
        # Potential energy update
            self._set_energy()
        # Gradient Update
            self._set_gradient()
        # velocities update
            self._update_velocities()
        # Kinetic update
            self._set_kinetic()
        # Temperature update
            self._set_instant_temp()
            # if FIT_VAR_LOCAL in self.vars:
            #     self._set("local_v", self._get("local_v") * (self.params["temperature"]/self._get("T")))

        # criterion update
            self._set_criterion()
            self.fit["L"][-1] +=1
        # Prints
            self._add("Time", time.time() -tt)
            if self.verbose > 1:
                self._print_step()
        if self.verbose == 1:
            self._print_step()
    # Hamiltonian update
        H = self._get_hamiltonian()
    # Metropolis acceptation
        self._acceptation(H, H_init)
    # save pdb step
        if self.prefix is not None:
            cp = self.init.copy()
            cp.coords = self._get("coord")
            cp.save_pdb(file=self.prefix+"_chain"+str(self.chain_id)+".pdb")
            del cp
            self.show(save=self.prefix+"_chain"+str(self.chain_id)+".png")

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

    def _set_factor(self, init_factor=100, **kwargs):
        psim = src.density.Volume.from_coords(coord=self.init.coords, size=self.target.size,
                                                 voxel_size=self.target.voxel_size,
                                                 sigma=self.target.sigma, cutoff=self.target.cutoff)
        U_biased = src.functions.get_RMSD(psim=psim.data, pexp=self.target.data)

        U_potential = src.forcefield.get_energy(coords=self.init.coords, forcefield=self.init.forcefield, **kwargs)["total"]
        factor = np.abs(init_factor/(U_biased/U_potential))
        if self.verbose > 0 : print("optimal initial factor : "+str(factor))
        return factor



def multiple_fitting(models, n_chain, n_proc):
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
            print("Failed to run multiple fitting : " + str(rte.args))

        print("\t\t done : "+str(time.time()-t))
    return fits






