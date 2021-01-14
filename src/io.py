import numpy as np
import src.molecule
import mrcfile

def read_pdb(file):
    atoms = []
    with open(file, "r") as file :
        end=False
        for line in file:
            l = line.split()
            if len(l) >0:
                if l[0] == 'ATOM':
                    if not end:
                        atoms.append(src.molecule.Atom(coord = [float(l[6]), float(l[7]), float(l[8])], type=l[2]))
                if l[0] == 'TER':
                    end = True

    return src.molecule.Molecule(atoms)

def save_pdb(mol, file, gen_file):
    print("Saving pdb file ...")
    with open(file, "w") as file:
        with open(gen_file, "r") as gen_file:
            n=0
            end = False
            for line in gen_file:
                split_line= line.split()

                if len(split_line) > 0:
                    if split_line[0] == 'ATOM':
                        l = [line[:6], line[6:11], line[12:16], line[17:20], line[21], line[22:26], line[30:38],
                             line[38:46], line[46:54], line[54:60], line[60:66], line[66:78]]
                        if split_line[2] == 'CA':
                            if not end:
                                coord = mol.atoms[n].coord
                                l[0] = l[0].ljust(6)  # atom#6s
                                l[1] = l[1].rjust(5)  # aomnum#5d
                                l[2] = l[2].center(4)  # atomname$#4s
                                l[3] = l[3].ljust(3)  # resname#1s
                                l[4] = l[4].rjust(1)  # Astring
                                l[5] = l[5].rjust(4)  # resnum
                                l[6] = str('%8.3f' % (float(coord[0]))).rjust(8)  # x
                                l[7] = str('%8.3f' % (float(coord[1]))).rjust(8)  # y
                                l[8] = str('%8.3f' % (float(coord[2]))).rjust(8)  # z\
                                l[9] = str('%6.2f' % (float(l[9]))).rjust(6)  # occ
                                l[10] = str('%6.2f' % (float(l[10]))).ljust(6)  # temp
                                l[11] = l[11].rjust(12)  # elname
                                file.write("%s%s %s %s %s%s    %s%s%s%s%s%s\n" % (l[0],l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], l[9], l[10], l[11]))
                                n+=1
                        if split_line[0] == 'TER':
                            file.write(line)
                            file.write("END\n")
                            end = True
    print("Done")

def save_density(density, outfilename):
    """
    Save the density to an mrc file. The origin of the grid will be (0,0,0)
    • outfilename: the mrc file name for the output
    """
    print("Saving mrc file ...")
    data = density.data.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data.T)
        mrc.voxel_size = density.sampling_rate
        origin = density.sampling_rate*density.n_voxels/2
        mrc.header['origin']['x'] = origin
        mrc.header['origin']['y'] = origin
        mrc.header['origin']['z'] = origin
        mrc.update_header_from_data()
        mrc.update_header_stats()
    print("Done")
