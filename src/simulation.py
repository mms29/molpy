import src.functions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Simulator:

    def __init__(self, atoms):
        self.init_structure=atoms
        self.n_atoms=atoms.shape[0]
        self.deformed_structure=None

    def run_nma(self, modes, amplitude=200):
        n_modes = modes.shape[1]
        self.q =np.random.uniform(-amplitude, amplitude, n_modes)
        if self.deformed_structure is None:
            init_structure = self.init_structure
            self.deformed_structure = np.zeros(self.init_structure.shape)
        else:
            init_structure=self.deformed_structure

        for i in range(self.n_atoms):
            self.deformed_structure[i]= np.dot(self.q, modes[i]) + init_structure[i]
        return self.deformed_structure


    def run_md(self, U_lim, step, bonds=None, angles=None, lennard_jones=None):
        if self.deformed_structure is None:
            init_structure = self.init_structure
        else:
            init_structure = self.deformed_structure

        deformed_structure = init_structure

        self.U_lim = U_lim
        U_init=0
        if bonds is not None:
            self.bonds_r0= np.mean([np.linalg.norm(self.init_structure[i] - self.init_structure[i + 1]) for i in range(self.n_atoms-1)])
            self.bonds_k = bonds['k']
            U_init += src.functions.md_bonds_potential(init_structure,self.bonds_k ,self.bonds_r0)
        if angles is not None:
            theta0 = []
            for i in range(self.n_atoms - 2):
                theta0.append(np.arccos(np.dot(self.init_structure[i] - self.init_structure[i + 1], self.init_structure[i + 1] - self.init_structure[i + 2])
                                           / (np.linalg.norm(self.init_structure[i] -self.init_structure[i + 1]) * np.linalg.norm(self.init_structure[i + 1] - self.init_structure[i + 2]))))
            self.angles_theta0 = np.mean(theta0)
            self.angles_k= angles['k']
            U_init += src.functions.md_angles_potential(init_structure, self.angles_k, self.angles_theta0)
        if lennard_jones is not None:
            self.lennard_jones_k = lennard_jones ['k']
            self.lennard_jones_d =  lennard_jones ['d']
            U_init += src.functions.md_lennard_jones_potential(init_structure,self.lennard_jones_k, self.lennard_jones_d)
        print("U init : " + str(U_init))

        self.md_variance = 0

        while (U_init > self.U_lim):

            # new direction
            dt = np.random.normal(0, step, init_structure.shape)
            deformed_structure = dt + init_structure

            U_deformed = 0
            if bonds is not None:
                U_bonds = src.functions.md_bonds_potential(deformed_structure, self.bonds_k, self.bonds_r0)
                U_deformed +=U_bonds
            if angles is not None:
                U_angles = src.functions.md_angles_potential(deformed_structure, self.angles_k,self.angles_theta0)
                U_deformed += U_angles
            if lennard_jones is not None:
                U_lennard_jones = src.functions.md_lennard_jones_potential(deformed_structure, self.lennard_jones_k, self.lennard_jones_d)
                U_deformed += U_lennard_jones


            if (U_init > U_deformed):
                U_init = U_deformed
                init_structure = deformed_structure
                self.md_variance += np.var(dt)
                s="U="+str(U_deformed)+" : "
                if bonds is not None:
                    s += " U_bonds="+str(U_bonds)+"  ; "
                if angles is not None:
                    s += " U_angles="+str(U_angles)+"  ; "
                if lennard_jones is not None:
                    s += " U_lennard_jones="+str(U_lennard_jones)+"  ; "
                print(s)

        self.deformed_structure = deformed_structure
        if self.md_variance==0 : self.md_variance=1
        return self.deformed_structure

    def compute_density(self, size=64,  sigma=1, sampling_rate=1):
        self.init_density= src.functions.volume_from_pdb(self.init_structure, size, sigma=sigma, sampling_rate=sampling_rate, precision=0.0001)
        self.deformed_density = src.functions.volume_from_pdb(self.deformed_structure, size, sigma=sigma, sampling_rate=sampling_rate, precision=0.0001)
        self.n_voxels=size

    def rotate_pdb(self):
        if self.deformed_structure is None:
            init_structure = self.init_structure
        else:
            init_structure = self.deformed_structure

        deformed_structure = init_structure

        alpha = np.random.uniform(0, 2*np.pi)
        beta = np.random.uniform(0, np.pi)
        gamma = np.random.uniform(0, 2*np.pi)

        R = np.array([[np.cos(gamma) * np.cos(alpha) * np.cos(beta) - np.sin(gamma) * np.sin(alpha), np.cos(gamma) * np.cos(beta)*np.sin(alpha) + np.sin(gamma) * np.cos(alpha), -np.cos(gamma) * np.sin(beta)],
                   [-np.sin(gamma) * np.cos(alpha) * np.cos(beta) - np.cos(gamma) * np.sin(alpha),-np.sin(gamma) * np.cos(beta)*np.sin(alpha) + np.cos(gamma) * np.cos(alpha), np.sin(gamma) * np.sin(beta)],
                   [np.sin(beta)*np.cos(alpha),np.sin(beta)*np.sin(alpha),np.cos(beta)]])

        rotated_atoms = np.zeros(self.init_structure.shape)
        for i in range(self.n_atoms):
            deformed_structure[i] = np.dot(init_structure[i], R)

        self.deformed_structure = deformed_structure
        return self.deformed_structure

    def plot_structure(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        legend=["init_structure"]
        ax.plot(self.init_structure[:, 0], self.init_structure[:, 1], self.init_structure[:, 2])
        if self.deformed_structure is not None:
            ax.plot(self.deformed_structure[:, 0], self.deformed_structure[:, 1], self.deformed_structure[:, 2])
            legend.append("deformed_structure")
        ax.legend(legend)
        # fig.show()

    def plot_density(self):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.init_density[int(self.n_voxels/2)], cmap='gray')
        ax[0].set_title("init_structure")
        ax[1].imshow(self.deformed_density[int(self.n_voxels/2)], cmap='gray')
        ax[1].set_title("deformed_structure")
        # fig.show()


