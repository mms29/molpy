import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import src.constants
import src.io
import os

def structures_viewer(structures, names=None, save=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    legend=[]
    if isinstance(structures, list):
        for i in range(len(structures)):
            coord=  structures[i].coords
            ax.plot(coord[:, 0], coord[:, 1], coord[:, 2])
            if names is not None:
                legend.append(names[i])
    else:
        for j in range(structures.n_chain):
            coord = structures.get_chain(j)
            ax.plot(coord[:, 0], coord[:, 1], coord[:, 2])
        if names is not None:
            legend.append(names)
    ax.legend(legend)
    if save is not None:
        fig.savefig(save)

# def density_viewer(density):
#     fig, ax = plt.subplots(1, 1)
#     axcolor = 'lightgoldenrodyellow'
#     axslider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
#     slice = Slider(axslider, 'Slice', valmin=0, valmax=density.size-1, valinit=int(density.size / 2), valstep=1)
#     ax.imshow(density.data[int(int(density.size / 2))], cmap='gray')
#
#     def update (val):
#         s = slice.val
#         ax.imshow(density.data[int(val)], cmap='gray')
#         fig.canvas.draw_idle()
#     slice.on_changed(update)


def image_viewer(img):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img.data, cmap='gray')

def internal_viewer(mol):
    fig, ax = plt.subplots(1, 3, figsize=(12,5))
    ax[0].set_title("Bonds")
    h = ax[0].hist(mol.bonds,100)
    ax[0].set_xlim(np.min(mol.bonds), np.max(mol.bonds))
    x=np.arange(np.min(mol.bonds), np.max(mol.bonds), 0.01)
    fx = np.square(x-src.constants.R0_BONDS)
    fx *= np.max(h[0])/np.max(fx)
    ax[0].plot(x, fx)

    ax[1].set_title("Angles")
    h = ax[1].hist(mol.angles,100)
    ax[1].set_xlim(0, np.pi/2)
    x = np.arange(0, np.pi/2, 0.01)
    fx = np.square(x - src.constants.THETA0_ANGLES)
    fx *= np.max(h[0]) / np.max(fx)
    ax[1].plot(x, fx)

    ax[2].set_title("Torsions")
    h = ax[2].hist(mol.torsions,100)
    ax[2].set_xlim(- np.pi, np.pi)
    x = np.arange(- np.pi, np.pi, 0.01)
    fx = 1+ np.cos(src.constants.N_TORSIONS*x - src.constants.DELTA_TORSIONS)
    fx *= np.max(h[0]) / np.max(fx)
    ax[2].plot(x, fx)

def density_viewer(density):
    def process_key(event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            previous_slice(ax)
        elif event.key == 'k':
            next_slice(ax)
        fig.canvas.draw()

    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def previous_slice(ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])

    def next_slice(ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])

    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    volume= density.data
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index], cmap='gray')
    fig.canvas.mpl_connect('key_press_event', process_key)

def chimera_fit_viewer(mol, target, genfile):
    src.io.save_pdb(mol, "mol.pdb" , genfile)
    src.io.save_density(target, "vol.mrc")
    cmd = "~/scipion3/software/em/chimerax-1.1/bin/ChimeraX --cmd \"open mol.pdb ; open vol.mrc ; volume #2 level 0.7 ; volume #2 transparency 0.7 ; hide atoms ; show cartoons\""
    os.system(cmd)

def chimera_structure_viewer(mol, genfile):
    if isinstance(mol, list):
        if not isinstance(genfile, list) :
            genfile = [genfile for i in range(len(mol))]
        cmd = "~/scipion3/software/em/chimerax-1.1/bin/ChimeraX --cmd \""
        for i in range(len(mol)):
            src.io.save_pdb(mol[i], "mol"+str(i)+".pdb", genfile[i])
            cmd+= "open mol"+str(i)+".pdb ; "
        cmd+="hide atoms ; show cartoons\""
    else:
        src.io.save_pdb(mol, "mol.pdb", genfile)
        cmd = "~/scipion3/software/em/chimerax-1.1/bin/ChimeraX --cmd \"open mol.pdb ; hide atoms ; show cartoons\""
    os.system(cmd)
