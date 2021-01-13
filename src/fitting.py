import src.functions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
import time
import pickle

class Fitting:

    def __init__(self, input_data, model_name, build=False):
        self.model = src.functions.read_stan_model(model_name, build=build)
        self.input_data = input_data
        self.sampling_results=None
        self.opt_results=None
        self.vb_results=None

    def sampling(self, n_iter, n_warmup, n_chain):
        t = time.time()
        self.fit = self.model.sampling(data=self.input_data, iter=n_iter+n_warmup, warmup=n_warmup, chains=n_chain)
        self.sampling_time = time.time() - t
        self.sampling_results = self.fit.extract(permuted=True)
        print("### EXECUTION ENDED "+ str(self.sampling_time)+"s ###")

    def optimizing(self,n_iter):
        t = time.time()
        try :
            self.opt_results= self.model.optimizing(data = self.input_data,iter=n_iter)
        except RuntimeError:
            try:
                self.opt_results = self.model.optimizing(data=self.input_data, iter=n_iter)
            except RuntimeError:
                self.opt_results = self.model.optimizing(data=self.input_data, iter=n_iter)
        self.opt_time = time.time() - t
        print("### EXECUTION ENDED " + str(self.opt_time) + "s ###")

    def vb(self,n_iter):
        t = time.time()
        self.vb_results= self.model.vb(data = self.input_data, iter=n_iter)
        self.vb_time = time.time() - t
        print("### EXECUTION ENDED " + str(self.vb_time) + "s ###")

    def read_results(self, type, param):
        if type=="sampling":
            return self.sampling_results[param]
        elif type=="opt":
            return self.opt_results[param]
        elif type=="vb":
            tmp =np.where(True== np.array([i.startswith(param+"[") for i in self.vb_results["sampler_param_names"]]))[0]
            start = np.min(tmp)
            stop = np.max(tmp)
            size_param = list(map(int, re.findall(r'\d+', self.vb_results["sampler_param_names"][stop])))
            n_iter =len(self.vb_results["sampler_params"][0])
            p = np.zeros([1000]+size_param)
            for i in range(size_param[0]):
                if len(size_param)>1:
                    for j in range(size_param[1]):
                        p[:,i,j] = self.vb_results["sampler_params"][start+ j*size_param[0]+ i]
                else:
                    p[:, i] = self.vb_results["sampler_params"][i+start]
            return p

    def test_grad(self):
        self.model.sampling(data=self.input_data, test_grad=True)

    def plot_nma(self, q_sim=None, save=None):
        n_modes = self.input_data['n_modes']
        legend=[]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(n_modes):
            if self.sampling_results is not None:
                parts = ax.violinplot(self.read_results("sampling", "q")[:, i], [i], )
                for pc in parts['bodies']:
                    pc.set_facecolor('tab:blue')
                    pc.set_edgecolor('grey')
                for partname in ('cbars', 'cmins', 'cmaxes',):
                    vp = parts[partname]
                    vp.set_edgecolor('tab:blue')
            if self.vb_results is not None:
                parts = ax.violinplot(self.read_results("vb", "q")[:, i], [i], )
                for pc in parts['bodies']:
                    pc.set_facecolor('tab:green')
                    pc.set_edgecolor('grey')
                for partname in ('cbars', 'cmins', 'cmaxes',):
                    vp = parts[partname]
                    vp.set_edgecolor('tab:green')
            if q_sim is not None:
                ax.plot(i, q_sim[i], 'x', color='r')
            if self.opt_results is not None:
                ax.plot(i, self.read_results("opt", "q")[i], 'x', color='tab:orange')

        ax.set_xlabel("modes")
        ax.set_ylabel("amplitude")
        ax.legend(legend)
        fig.suptitle("Normal modes amplitudes distribution")
        if save is not None : fig.savefig(save)
        # fig.show()

    def plot_lp(self, save=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        lp = self.fit.extract(pars='lp__',inc_warmup=True, permuted=False)['lp__']
        llp = np.sign(lp) * np.log(1 + np.abs(lp))
        for i in range(lp.shape[1]):
            ax.plot(llp[:,i])
        if save is not None: fig.savefig(save)

    def plot_structure(self, other_structure=None, save=None):
        if 'x0' in self.input_data:
            init_structure =self.input_data['x0']
        else:
            internal = np.array([self.input_data['bonds'],self.input_data['angles'],self.input_data['torsions']]).T
            init_structure = src.functions.internal_to_cartesian(internal, self.input_data['first'])
        target_structure = self.input_data['y']
        legend=["init_structure","target_structure"]
        structures = [init_structure, target_structure]

        if self.sampling_results is not None:
            structures.append(np.mean(self.read_results("sampling", "x"), axis=0))
            legend.append("sampling_structure")
        if self.opt_results is not None:
            structures.append(self.read_results("opt", "x"))
            legend.append("opt_structure")
        if self.vb_results is not None:
            structures.append(np.mean(self.read_results("vb", "x"), axis=0))
            legend.append("vb_structure")
        if other_structure is not None:
            structures.append(other_structure)
            legend.append("other_structure")
        src.functions.plot_structure(structures, legend, save)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(obj=self, file=f)

    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as f:
            return pickle.load(file=f)

    def plot_error_map(self, N=64, sigma=2, sampling_rate=4, save=None, figsize=(10,5), slice=None):
        if slice is None:
            slice = int(N/2)
        if ("em_density" in self.input_data):
            em_density_target=self.input_data["em_density"]
        else:
            em_density_target = src.functions.volume_from_pdb(self.input_data["y"], N, sigma=sigma, sampling_rate=sampling_rate)
        if 'x0' in self.input_data:
            init_structure =self.input_data['x0']
        else:
            internal = np.array([self.input_data['bonds'],self.input_data['angles'],self.input_data['torsions']]).T
            init_structure = src.functions.internal_to_cartesian(internal, self.input_data['first'])
        em_density_init = src.functions.volume_from_pdb(init_structure, N, sigma=sigma,
                                                          sampling_rate=sampling_rate)
        n_plot =0
        if self.vb_results is not None: n_plot+=1
        if self.opt_results is not None: n_plot += 1
        if self.sampling_results is not None: n_plot += 1

        fig, ax = plt.subplots(1, n_plot+1, figsize=figsize)
        n_plot=0
        err_map_init = np.square(em_density_init - em_density_target)[slice]
        ax[n_plot].imshow(err_map_init, vmax=np.max(err_map_init), cmap='jet')
        ax[n_plot].set_title("init : e=%.2g" % src.functions.root_mean_square_error(em_density_init,em_density_target))
        if self.vb_results is not None:
            n_plot+=1
            em_density_vb = src.functions.volume_from_pdb(np.mean(self.read_results("vb", "x"), axis=0), N, sigma=sigma,
                                                          sampling_rate=sampling_rate)
            err_map_vb = np.square(em_density_target - em_density_vb)[slice]
            ax[n_plot].imshow(err_map_vb, vmax=np.max(err_map_init), cmap='jet')
            ax[n_plot].set_title("vb : e=%.2g" % src.functions.root_mean_square_error(em_density_vb,em_density_target))

        if self.opt_results is not None:
            n_plot+=1
            em_density_opt = src.functions.volume_from_pdb(self.read_results("opt", "x"), N, sigma=sigma,
                                                          sampling_rate=sampling_rate)
            err_map_opt = np.square(em_density_target - em_density_opt)[slice]
            ax[n_plot].imshow(err_map_opt, vmax=np.max(err_map_init), cmap='jet')
            ax[n_plot].set_title("opt : e=%.2g" % src.functions.root_mean_square_error(em_density_opt,em_density_target))
        if self.sampling_results is not None:
            n_plot+=1
            em_density_sampling = src.functions.volume_from_pdb(np.mean(self.read_results("sampling", "x"), axis=0), N, sigma=sigma,
                                                          sampling_rate=sampling_rate)
            err_map_sampling = np.square(em_density_target - em_density_sampling)[slice]
            ax[n_plot].imshow(err_map_sampling, vmax=np.max(err_map_init), cmap='jet')
            ax[n_plot].set_title("sampling : e=%.2g" % src.functions.root_mean_square_error(em_density_sampling,em_density_target))
        if save is not None : fig.savefig(save)