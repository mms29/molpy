import src.constants
import src.force_field
import src.molecule
import numpy as np


def run_md(mol, dt, total_time, temperature=300):

    n_steps = int(total_time/dt)
    xt = mol.coords
    vt = np.random.normal(0, np.sqrt(src.constants.K_BOLTZMANN*temperature / src.constants.CARBON_MASS), xt.shape)
    Ft = src.force_field.get_gradient(mol)

    molstep=[mol]
    Fstep=[Ft]

    for i in range(n_steps):
        print("ITER = "+str(i))

        # positions updates
        xt = xt + dt*vt + dt**2 *Ft/2

        # velocities update
        molt = src.molecule.Molecule.from_coords(xt)
        Ftt = src.force_field.get_gradient(molt)
        vt = vt * dt*(Ft +Ftt)/2
        Ft = Ftt

        molstep.append(molt)
        Fstep.append(Ftt)

    return molstep, Fstep
