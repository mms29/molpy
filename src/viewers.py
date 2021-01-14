import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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
    slice = Slider(axslider, 'Slice', valmin=0, valmax=density.n_voxels-1, valinit=int(density.n_voxels / 2), valstep=1)
    ax.imshow(density.data[int(int(density.n_voxels / 2))], cmap='gray')

    def update (val):
        s = slice.val
        ax.imshow(density.data[int(val)], cmap='gray')
        fig.canvas.draw_idle()
    slice.on_changed(update)

