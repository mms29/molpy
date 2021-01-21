import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import src.constants

def structures_viewer(structures, names=None, save=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    legend=[]
    if isinstance(structures, list):
        for i in range(len(structures)):
            ax.plot(structures[i].coords[:, 0], structures[i].coords[:, 1], structures[i].coords[:, 2])
            if names is not None:
                legend.append(names[i])
    else:
        ax.plot(structures.coords[:, 0], structures.coords[:, 1], structures.coords[:, 2])
        if names is not None:
            legend.append(names)
    ax.legend(legend)
    if save is not None:
        fig.savefig(save)

def density_viewer(density):
    fig, ax = plt.subplots(1, 1)
    axcolor = 'lightgoldenrodyellow'
    axslider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    slice = Slider(axslider, 'Slice', valmin=0, valmax=density.size-1, valinit=int(density.size / 2), valstep=1)
    ax.imshow(density.data[int(int(density.size / 2))], cmap='gray')

    def update (val):
        s = slice.val
        ax.imshow(density.data[int(val)], cmap='gray')
        fig.canvas.draw_idle()
    slice.on_changed(update)


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

