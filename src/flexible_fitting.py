import src.constants
from src.molecule import *
from src.density import *

import numpy as np
from src.functions import *
from src.forcefield import *
import time
import pickle
from multiprocessing import Pool
import mkl
mkl.set_num_threads(1)

class FlexibleFitting:
    def __init__(self, init, target, vars, n_chain, n_iter, n_warmup, params, verbose=0):
        self.init = init
        self.target = target
        self.params = params
        self.n_chain = n_chain
        self.n_iter = n_iter
        self.n_warmup= n_warmup
        self.verbose = verbose
        self.vars = vars

    def HMC(self):
        with Pool(self.n_chain) as p:
            fittings = p.map(FlexibleFitting.HMC_chain, [self for i in range(self.n_chain)])
            p.close()
            p.join()

        self.res = {"mol": Molecule.from_molecule(self.init)}
        self.res["mol"].coords = np.mean([fittings[i].res["mol"].coords for i in range(self.n_chain)], axis=0)
        for v in self.vars:
            self.res[v] = np.mean([fittings[i].res[v] for i in range(self.n_chain)], axis=0)
        self.fit = [fittings[i].fit for i in range(self.n_chain)]
        return self

    def HMC_chain(self):
        np.random.seed()
        self.fit= {"coord":[copy.deepcopy(self.init.coords)]}
        for i in self.vars:
            self._set(i ,[self.params[i+"_init"]])

        # HMC Loop
        for i in range(self.n_iter):
            if self.verbose > 0 : print("HMC ITER = " + str(i))
            self.HMC_step()

        self.res = {"mol" : Molecule.from_molecule(self.init)}
        self.res["mol"].coords = np.mean(np.array(self.fit["coord"][self.n_warmup + 1:]), axis=0)

        for i in self.vars:
            self.res[i] = np.mean(np.array(self.fit[i][self.n_warmup+1:]), axis=0)

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
        U = 0

        U_biased = src.functions.get_RMSD(psim=psim, pexp=self.target.data) * self.params["lb"]
        self._add("U_biased", U_biased)
        U+= U_biased

        U_potential = get_energy(self.init, self._get("coordt"), verbose=False)* self.params["lp"]
        self._add("U_potential", U_potential)
        U += U_potential

        for i in self.vars:
            if "l"+i in self.params:
                U_var = np.sum(np.square(self._get(i+"_t"))) * self.params["l"+i]
                self._add("U_"+i, U_var)
                U += U_var

        self._add("U", U)

    def _set_gradient(self, psim):
        vals={}
        for i in self.vars:
            vals[i] = self._get(i+"_t")

        dU_biased = self.target.get_gradient_RMSD(self.init, psim, vals)
        dU_potential = get_autograd(self.init, vals)

        for i in self.vars:
            F = -((self.params["lb"] * dU_biased[i]) + (self.params["lp"] *  dU_potential[i]))
            if "l" + i in self.params:
                F+=  - 2* self._get(i+"_t") * self.params["l"+i]

            if not i+"_F" in self.fit: self._set(i+"_F", F)
            else: self._set(i+"_Ft", F)

    def _set_kinetic(self):
        K=0
        for i in self.vars:
            K += 1 / 2 * np.sum(self.params[i+"_mass"]*np.square(self._get(i+"_t")))
        self._add("K", K)

    def _set_criterion(self):
        C = 0
        if self.params["criterion"]:
            for i in self.vars:
                C += np.dot((self._get(i+"_t").flatten() - self._get(i).flatten()), self._get(i+"_v").flatten())
            self._add("C",C)
        else:
            self._add("C", 0)

    def _set_density(self):
        if isinstance(self.target, Volume):
            return Volume.from_coords(coord=self._get("coordt"), size=self.target.size,
                                      voxel_size=self.target.voxel_size,
                                      sigma=self.target.sigma, threshold=self.target.threshold).data
        else:
            return Image.from_coords(coord=self._get("coordt"), size=self.target.size,
                                      voxel_size=self.target.voxel_size,
                                      sigma=self.target.sigma, threshold=self.target.threshold).data


    def _print_step(self):
        s=["L", "U_biased", "U_potential", "CC", "Time", "K"]
        for i in self.vars :
            if "l"+i in self.params:
                s.append("U_"+i)
        printed_string=""
        for i in s:
            printed_string += i + "=" + str(self._get(i)) + " ; "
        print(printed_string)

    def _initialize(self):
        for i in self.vars:
            self._set(i+"_t", self._get(i))
            self._set(i+"_v", np.random.normal(0, self.params[i+"_mass"], self._get(i).shape))

    def _update_positions(self):
        for i in self.vars:
            self._set(i+"_t", self._update_pstep(self._get(i + "_t"), self._get(i + "_v"), self.params[i + "_dt"],
                                               self._get(i + "_F")))

    def _forward_model(self):
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


    def _update_pstep(self, x, v, dx, F):
        return x+ dx*v + dx**2 *(F/2)

    def _update_velocities(self):
        for i in self.vars:
            self._set(i+"_v", self._update_vstep(self._get(i+"_v") , self.params[i+"_dt"] , self._get(i+"_F") ,self._get(i+"_Ft")))
            self._set(i+"_F" ,self._get(i+"_Ft"))

    def _update_vstep(self, v, dx, F, Ft):
        return v + dx*((F+Ft)/2)

    def _acceptation(self,H, H_init):
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

    def HMC_step(self):

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
        if isinstance(self.fit, list):
            fits= self.fit
        else :
            fits= [self.fit]
        fig, ax = plt.subplots(2, 3, figsize=(10, 5))
        for i in range(self.n_chain):
            ax[0, 0].plot(fits[i]['U'])
            ax[0, 1].plot(fits[i]['U_potential'])
            ax[0, 2].plot(fits[i]['U_biased'])
            ax[1, 0].plot(np.array(fits[i]['K']) + np.array(fits[i]['U']))
            ax[1, 0].plot(fits[i]['U'])
            ax[1, 0].plot(fits[i]['K'])
            ax[1, 1].plot(fits[i]['C'])
            ax[1, 2].plot(fits[i]['CC'])
        ax[0, 0].set_title('U')
        ax[0, 1].set_title('U_potential')
        ax[0, 2].set_title('U_biased')
        ax[1, 0].set_title('H=U+K')
        ax[1, 1].set_title('C')
        ax[1, 2].set_title('CC')
        fig.tight_layout()

    def show_3D(self):
        src.viewers.chimera_fit_viewer(self.res["mol"], self.target)

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(file=f, obj=self)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            fit = pickle.load(file=f)
            return fit






