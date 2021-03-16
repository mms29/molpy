import time
import multiprocessing
import multiprocessing.pool
from multiprocessing import Pool
import copy
import pickle

import src.constants
import src.density
import src.forcefield
from src.functions import *
import src.molecule

class FlexibleFitting:
    """
    Perform flexible fitting between initial atomic structure and target Density using HMC
    """

    def __init__(self, init, target, vars, n_chain, n_iter, n_warmup, params, verbose=0):
        """
        Constructor
        :param init: inititial atomic structure Molecule
        :param target: target Density
        :param vars: list of fitted variables
        :param n_chain: number of chain
        :param n_iter: number of iterations
        :param n_warmup: number of warmup iterations
        :param params: fit parameters
        :param verbose: verbose level
        """
        self.init = init
        self.target = target
        self.params = params
        self.n_chain = n_chain
        self.n_iter = n_iter
        self.n_warmup= n_warmup
        self.verbose = verbose
        self.vars = vars

    def HMC(self):
        """
        Run HMC fitting with the specified number of chain in parallel
        :return: fit
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
        :return: fit
        """

        # set the random seed of numpy for parallel computation
        np.random.seed()
        t = time.time()

        # initialize fit variables
        self.fit= {"coord":[copy.deepcopy(self.init.coords)]}
        for i in self.vars:
            self._set(i ,[self.params[i+"_init"]])

        # HMC Loop
        for i in range(self.n_iter):
            if self.verbose > 0 : print("HMC ITER = " + str(i))
            self.HMC_step()

        # Generate results
        self.res = {"mol" : src.molecule.Molecule.from_molecule(self.init)}
        self.res["mol"].coords = np.mean(np.array(self.fit["coord"][self.n_warmup + 1:]), axis=0)
        for i in self.vars:
            self.res[i] = np.mean(np.array(self.fit[i][self.n_warmup+1:]), axis=0)

        # End
        if self.verbose >0:
            print("############### HMC FINISHED ##########################")
            print("### Total execution time : "+str(time.time()-t)+" s")
            print("### Initial CC value : "+str(self.fit["CC"][0]))
            print("### Mean CC value : "+str(np.mean(self.fit["CC"][self.n_warmup:])))
            print("#######################################################")

        return self

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

        # Biased Potential
        U_biased = src.functions.get_RMSD(psim=psim, pexp=self.target.data) * self.params["lb"]
        self._add("U_biased", U_biased)
        U+= U_biased

        # Energy Potential
        U_potential = src.forcefield.get_energy(coord=self._get("coordt"), molstr = self.init.psf,
                                 molprm= self.init.prm, verbose=False)* self.params["lp"]
        self._add("U_potential", U_potential)
        U += U_potential

        # Additional Priors on parameters
        for i in self.vars:
            if "l"+i in self.params:
                U_var = np.sum(np.square(self._get(i+"_t"))) * self.params["l"+i]
                self._add("U_"+i, U_var)
                U += U_var

        # Total energy
        self._add("U", U)

    def _set_gradient(self, psim):
        """
        Compute the gradient of the total energy from the simulated Density
        :param psim: simulated Density
        """
        vals={}
        for i in self.vars:
            vals[i] = self._get(i+"_t")

        dU_biased = self.target.get_gradient_RMSD(self.init, psim, vals)
        dU_potential = src.forcefield.get_autograd(params=vals, mol = self.init)

        for i in self.vars:
            F = -((self.params["lb"] * dU_biased[i]) + (self.params["lp"] *  dU_potential[i]))
            if "l" + i in self.params:
                F+=  - 2* self._get(i+"_t") * self.params["l"+i]

            if not i+"_F" in self.fit: self._set(i+"_F", F)
            else: self._set(i+"_Ft", F)

    def _set_kinetic(self):
        """
        Compute the Kinetic energy
        """
        K=0
        for i in self.vars:
            K += 1 / 2 * np.sum(self.params[i+"_mass"]*np.square(self._get(i+"_t")))
        self._add("K", K)

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
        if isinstance(self.target, src.density.Volume):
            return src.density.Volume.from_coords(coord=self._get("coordt"), size=self.target.size,
                                      voxel_size=self.target.voxel_size,
                                      sigma=self.target.sigma, threshold=self.target.threshold).data
        else:
            return src.density.Image.from_coords(coord=self._get("coordt"), size=self.target.size,
                                      voxel_size=self.target.voxel_size,
                                      sigma=self.target.sigma, threshold=self.target.threshold).data

    def _initialize(self):
        """
        Initialize all variables positions and velocities
        """
        for i in self.vars:
            self._set(i+"_t", self._get(i))
            self._set(i+"_v", np.random.normal(0, self.params[i+"_mass"], self._get(i).shape))

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
        coordt = copy.deepcopy(self.init.coords)
        if "x" in self.vars:
            coordt += self._get("x_t")
        if "q" in self.vars:
            coordt += np.dot(self._get("q_t"), self.init.modes)
        if "angles" in self.vars:
            coordt = np.dot( src.functions.generate_euler_matrix(self._get("angles_t")),  coordt.T).T
        if "shift"  in self.vars:
            coordt += self._get("shift_t")
        self._set("coordt", coordt)

    def _acceptation(self,H, H_init):
        """
        Perform Metropolis Acceptation step
        :param H: Current Hamiltonian
        :param H_init: Initial Hamiltonian
        """
        # accept_p = np.min([1, np.exp((H_init - H))])
        accept_p = np.min([1, np.exp((H_init - H)/H_init)])
        if self.verbose > 1 : print("H<H_init="+str(H_init > H) +" ; H" + str(H)+" ; H_init" + str(H_init))
        if accept_p > np.random.rand() :
            if self.verbose > 1 : print("ACCEPTED " + str(accept_p))
            for i in self.vars:
                self._add(i, self._get(i+"_t"))
            self._add("coord", self._get("coordt"))
        else:
            if self.verbose > 1 : print("REJECTED " + str(accept_p))
            for i in self.vars:
                self._add(i, self._get(i))
            self._add("coord", self._get("coord"))
        for i in self.vars:
            self._remove(i+"_F")

    def _print_step(self):
        """
        Print step information
        """
        s=["L", "U_biased", "U_potential", "CC", "Time", "K"]
        for i in self.vars :
            if "l"+i in self.params:
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
        H_init = self._get("U") + self._get("K")
    # Init vars
        self._add("C",0)
        self._add("L", 0)
    # MD loop
        while (self._get("C") >= 0 and self._get("L")< self.params["max_iter"]):
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
            T = 2 * self._get("K") / (src.constants.K_BOLTZMANN * 3* self.init.n_atoms)
        # criterion update
            self._set_criterion()
            self.fit["L"][-1] +=1
        # Prints
            self._add("Time", time.time() -tt)
            if self.verbose > 1:
                self._print_step()
    # Hamiltonian update
        H = self._get("U") + self._get("K")
    # Metropolis acceptation
        self._acceptation(H, H_init)

    def show(self):
        """
        Show fitting statistics
        """
        src.viewers.fit_viewer(self)

    def show_3D(self):
        """
        Show fitting results in 3D
        """
        src.viewers.chimera_fit_viewer(self.res["mol"], self.target)

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(file=f, obj=self)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            fit = pickle.load(file=f)
            return fit


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

        print("\t fitting models # "+str(i))
        try :
            with NestablePool(n_process) as p:
                fits += p.map(FlexibleFitting.HMC, models[i])
                p.close()
                p.join()
        except RuntimeError:
            print("Failed")
    return fits






