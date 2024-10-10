import numpy as np
import os
from ase import Atoms
from ase.io.cube import read_cube
from numpy.random import default_rng
from random import shuffle
rng = default_rng(42)

# read charge density / response file
def Read_data(folder, Prop_dict, grid_configuration, alpha, sigma):

    numpoint = [0, 0]
    filepath_list = [os.path.join(folder, file) for file in os.listdir(folder)]
    atoms_with_grid_points_list = list()
    charge_density_list = []
    for filepath in filepath_list:
        with open(filepath, "r") as fd:

            # read the .cube file
            charge_cube = read_cube(fd)
            atoms, charge = charge_cube["atoms"], charge_cube["data"]
            atoms.set_pbc([True, True, True])
            cell = np.array(atoms.get_cell())
            origin = np.zeros(3)
            ngridpts = np.array(charge.shape)  # grid matrix
            grid_pos = np.meshgrid(
                np.arange(ngridpts[0]) / charge.shape[0],
                np.arange(ngridpts[1]) / charge.shape[1],
                np.arange(ngridpts[2]) / charge.shape[2],
                indexing="ij",
            )
            grid_pos = np.stack(grid_pos, 3)
            grid_pos = np.dot(grid_pos, cell) + origin
            grid_space = np.diagonal(cell) / ngridpts

            # sampling gird points from entire mesh by density value and its gradient
            grad_charge = np.linalg.norm(np.array(np.gradient(charge)), axis=0).reshape(-1)
            grid_pos = grid_pos.reshape(-1, 3)
            charge = charge.reshape(-1)
            prob_grid = np.abs(charge) + alpha * grad_charge + 2e-5

            # prob_grid = np.exp(-((1 / prob_grid) ** 2) / (2 * sigma ** 2)) / ((2 * np.pi * (sigma ** 2)) ** 0.5)
            prob_grid /= sum(prob_grid)
            selected_index = rng.choice(np.arange(charge.shape[0]), size=grid_configuration, p=prob_grid, replace=False)
            probe_pos = grid_pos[selected_index].reshape(-1, 3)
            probe_target = charge[selected_index].reshape(-1)

            grid_batch = Prop_dict["ChargeDensity"]
            numbatch = int(np.ceil(grid_configuration / grid_batch))
            for ibatch in range(numbatch):
                down_index = grid_batch*ibatch
                up_index = np.min([grid_batch * (ibatch + 1), grid_configuration])
                grid_points = Atoms(
                    symbols=["X" for _ in range(down_index, up_index)],
                    positions=probe_pos[down_index:up_index, :],
                    masses=[0.0 for _ in range(down_index, up_index)],
                    cell=atoms.get_cell(),
                )
                atoms_with_grid_points_list.append(atoms+grid_points)
                charge_density_list.append(probe_target[down_index:up_index].tolist())
                # print(charge_density_list[-1])

    # export data format
    
    zip_data = list(zip(charge_density_list, atoms_with_grid_points_list))
    shuffle(zip_data)
    charge_density_list, atoms_with_grid_points_list = zip(*zip_data)
    numpoint[0] = len(atoms_with_grid_points_list)
    atom = [list(atoms.symbols) for atoms in atoms_with_grid_points_list]
    mass = [atoms.get_masses().tolist() for atoms in atoms_with_grid_points_list]
    numatoms = [len(atoms) for atoms in atoms_with_grid_points_list]
    scalmatrix = [atoms.get_cell().tolist() for atoms in atoms_with_grid_points_list]
    period_table = [atoms.get_pbc().tolist() for atoms in atoms_with_grid_points_list]
    coor = [atoms.get_positions().tolist() for atoms in atoms_with_grid_points_list]
    ef = [[0.0, 0.0, 1.0] for _ in atoms_with_grid_points_list]

    return numpoint, atom, mass, numatoms, scalmatrix, period_table, coor, ef, [charge_density_list,]


