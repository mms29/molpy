import src.functions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
import time
import pickle
from src.molecule import Molecule, Image, Density
import src.constants
import src.viewers

class Fitting:

    def __init__(self,init, target, model_name, build=False):
        self.model = src.functions.read_stan_model(model_name, build=build)
        self.target = target
        self.init = init
        self.results=None

    def sampling(self, n_iter, n_warmup, n_chain, param):
        t = time.time()
        self.param = param
        input_data = self.get_input_data(self.init, self.target, param)
        self.fit = self.model.sampling(data=input_data, iter=n_iter+n_warmup, warmup=n_warmup, chains=n_chain)
        self.fit_time = time.time() - t
        self.results = self.fit.extract(permuted=True)
        self.fitted_mol = Molecule(np.mean(self.results["x"],axis=0))
        print("### EXECUTION ENDED "+ str(self.fit_time)+"s ###")

        return self.fitted_mol

    def optimizing(self, n_iter, param):
        t = time.time()
        self.param = param
        input_data = self.get_input_data(self.init, self.target, param)
        self.results= self.model.optimizing(data = input_data,iter=n_iter)
        self.fit_time = time.time() - t
        self.fitted_mol = Molecule(self.results["x"])
        print("### EXECUTION ENDED " + str(self.fit_time) + "s ###")

        return self.fitted_mol

    def get_input_data(self, init, target, param):
        input_data = src.constants.DEFAULT_INPUT_DATA

        input_data["x0"] = init.coords
        input_data["n_atoms"] = init.n_atoms
        input_data["bonds"] = init.bonds[2:]
        input_data["angles"] = init.angles[1:]
        input_data["torsions"] = init.torsions
        input_data["first"] = init.coords[:3]
        if init.modes is not None :
            input_data["n_modes"]= init.modes.shape[1]
            input_data["A_modes"]= init.modes

        if isinstance(target, Molecule):
            input_data["epsilon"] = 1
            input_data["y"]  = target.coords
        else :
            input_data["epsilon"] = np.max(target.data)/10
            input_data["N"] = target.size
            input_data["sampling_rate"] = target.sampling_rate
            input_data["gaussian_sigma"] = target.gaussian_sigma
            input_data["halfN"] = int(target.size/2)
            input_data["density"] = target.data

        input_data["verbose"] = 0

        input_data.update(param)

        return input_data

    def plot_structure(self, target):
        if target is not None:
            src.viewers.structures_viewer([self.init, self.fitted_mol, target], names=["init", "fitted", "target"])
        else:
            src.viewers.structures_viewer([self.init, self.fitted_mol], names=["init", "fitted"])

    # def plot_nma(self, q_sim=None, save=None):
    #     n_modes = self.input_data['n_modes']
    #     legend=[]
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     for i in range(n_modes):
    #         if self.sampling_results is not None:
    #             parts = ax.violinplot(self.read_results("sampling", "q")[:, i], [i], )
    #             for pc in parts['bodies']:
    #                 pc.set_facecolor('tab:blue')
    #                 pc.set_edgecolor('grey')
    #             for partname in ('cbars', 'cmins', 'cmaxes',):
    #                 vp = parts[partname]
    #                 vp.set_edgecolor('tab:blue')
    #         if self.vb_results is not None:
    #             parts = ax.violinplot(self.read_results("vb", "q")[:, i], [i], )
    #             for pc in parts['bodies']:
    #                 pc.set_facecolor('tab:green')
    #                 pc.set_edgecolor('grey')
    #             for partname in ('cbars', 'cmins', 'cmaxes',):
    #                 vp = parts[partname]
    #                 vp.set_edgecolor('tab:green')
    #         if q_sim is not None:
    #             ax.plot(i, q_sim[i], 'x', color='r')
    #         if self.opt_results is not None:
    #             ax.plot(i, self.read_results("opt", "q")[i], 'x', color='tab:orange')
    #
    #     ax.set_xlabel("modes")
    #     ax.set_ylabel("amplitude")
    #     ax.legend(legend)
    #     fig.suptitle("Normal modes amplitudes distribution")
    #     if save is not None : fig.savefig(save)
    #     # fig.show()
    #
    def plot_lp(self, save=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        lp = self.fit.extract(pars='lp__',inc_warmup=True, permuted=False)['lp__']
        llp = np.sign(lp) * np.log(1 + np.abs(lp))
        for i in range(lp.shape[1]):
            ax.plot(llp[:,i])
        if save is not None: fig.savefig(save)
    #
    # def plot_structure(self, other_structure=None, save=None):
    #     pass


    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(obj=self, file=f)

    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as f:
            return pickle.load(file=f)

