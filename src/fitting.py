import src.functions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re

class Fitting:

    def __init__(self, input_data, model_name, build=False):
        self.model = src.functions.read_stan_model(model_name, build=build)
        self.input_data = input_data
        self.sampling_results=None
        self.opt_results=None
        self.vb_results=None

    def sampling(self, n_iter, n_warmup, n_chain):
        fit = self.model.sampling(data=self.input_data, iter=n_iter+n_warmup, warmup=n_warmup, chains=n_chain)
        self.sampling_results = fit.extract(permuted=True)

    def optimizing(self,n_iter):
        self.opt_results= self.model.optimizing(data = self.input_data,iter=n_iter)

    def vb(self,n_iter):
        self.vb_results= self.model.vb(data = self.input_data, iter=n_iter)

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

    def plot_nma(self, q_sim, save=None):
        n_modes = q_sim.shape[0]
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

    def plot_structure(self, other_structure=None, save=None):
        init_structure = self.input_data['x0']
        target_structure = self.input_data['y']
        legend=["init_structure","target_structure"]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(init_structure[:, 0], init_structure[:, 1], init_structure[:, 2])
        ax.plot(target_structure[:, 0], target_structure[:, 1], target_structure[:, 2])
        if self.sampling_results is not None:
            sampling_structure = np.mean(self.read_results("sampling", "x"), axis=0)
            ax.plot(sampling_structure[:, 0], sampling_structure[:, 1], sampling_structure[:, 2])
            legend.append("sampling_structure")
        if self.opt_results is not None:
            opt_structure = self.read_results("opt", "x")
            ax.plot(opt_structure[:, 0], opt_structure[:, 1], opt_structure[:, 2])
            legend.append("opt_structure")
        if self.vb_results is not None:
            vb_structure = np.mean(self.read_results("vb", "x"), axis=0)
            ax.plot(vb_structure[:, 0], vb_structure[:, 1], vb_structure[:, 2])
            legend.append("vb_structure")
        if other_structure is not None:
            ax.plot(other_structure[:, 0], other_structure[:, 1], other_structure[:, 2])
            legend.append("other_structure")
        ax.legend(legend)
        if save is not None: fig.savefig(save)
        # fig.show()

    def plot_error_map(self, N=64, sigma=2, sampling_rate=4, save=None, figsize=(10,5)):
        if ("em_density" in self.input_data):
            em_density_target=self.input_data["em_density"]
        else:
            em_density_target = src.functions.volume_from_pdb(self.input_data["y"], N, sigma=sigma, sampling_rate=sampling_rate, precision=0.0001)
        em_density_init = src.functions.volume_from_pdb(self.input_data["x0"], N, sigma=sigma,
                                                          sampling_rate=sampling_rate, precision=0.0001)
        n_plot =0
        if self.vb_results is not None: n_plot+=1
        if self.opt_results is not None: n_plot += 1
        if self.sampling_results is not None: n_plot += 1

        fig, ax = plt.subplots(1, n_plot+1, figsize=figsize)
        n_plot=0
        err_map_init = np.square(em_density_init - em_density_target)[int(N / 2)]
        ax[n_plot].imshow(err_map_init, vmax=np.max(err_map_init), cmap='jet')
        ax[n_plot].set_title("init : e=%.2g" % np.mean(err_map_init))
        if self.vb_results is not None:
            n_plot+=1
            em_density_vb = src.functions.volume_from_pdb(np.mean(self.read_results("vb", "x"), axis=0), N, sigma=sigma,
                                                          sampling_rate=sampling_rate, precision=0.0001)
            err_map_vb = np.square(em_density_target - em_density_vb)[int(N / 2)]
            ax[n_plot].imshow(err_map_vb, vmax=np.max(err_map_init), cmap='jet')
            ax[n_plot].set_title("vb : e=%.2g" % np.mean(err_map_vb))

        if self.opt_results is not None:
            n_plot+=1
            em_density_opt = src.functions.volume_from_pdb(self.read_results("opt", "x"), N, sigma=sigma,
                                                          sampling_rate=sampling_rate, precision=0.0001)
            err_map_opt = np.square(em_density_target - em_density_opt)[int(N / 2)]
            ax[n_plot].imshow(err_map_opt, vmax=np.max(err_map_init), cmap='jet')
            ax[n_plot].set_title("opt : e=%.2g" % np.mean(err_map_opt))
        if self.sampling_results is not None:
            n_plot+=1
            em_density_sampling = src.functions.volume_from_pdb(np.mean(self.read_results("sampling", "x"), axis=0), N, sigma=sigma,
                                                          sampling_rate=sampling_rate, precision=0.0001)
            err_map_sampling = np.square(em_density_target - em_density_sampling)[int(N / 2)]
            ax[n_plot].imshow(err_map_sampling, vmax=np.max(err_map_init), cmap='jet')
            ax[n_plot].set_title("sampling : e=%.2g" % np.mean(err_map_sampling))
        if save is not None : fig.savefig(save)