from pathlib import Path

import click
import numpy as np
import pandas as pd
from ase.io import read
from ase.spacegroup import get_spacegroup
from joblib import Parallel, delayed
from tqdm import tqdm

from statgen import to_format_table

OFFSET_LIST = np.array(
    [
        [-1, -1, -1],
        [-1, -1, 0],
        [-1, -1, 1],
        [-1, 0, -1],
        [-1, 0, 0],
        [-1, 0, 1],
        [-1, 1, -1],
        [-1, 1, 0],
        [-1, 1, 1],
        [0, -1, -1],
        [0, -1, 0],
        [0, -1, 1],
        [0, 0, -1],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, -1],
        [0, 1, 0],
        [0, 1, 1],
        [1, -1, -1],
        [1, -1, 0],
        [1, -1, 1],
        [1, 0, -1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, -1],
        [1, 1, 0],
        [1, 1, 1],
    ]
)


def get_rmsd(
    cart_coords1,
    cart_coords2,
    lattice,
):
    cart_coords1 = np.array(cart_coords1)
    cart_coords2 = np.array(cart_coords2)
    lattice = np.array(lattice)

    assert cart_coords1.shape == cart_coords2.shape, "coords shape not match"
    natoms = len(cart_coords1)
    num_cells = len(OFFSET_LIST)

    # offsets      (a,b,c)     (o1,o2,...o27)  -> (3,27) ((o1x,o2x, ...),(o1y,o2y,...),(o1z,o2z,...))
    pbc_offsets = lattice.T @ OFFSET_LIST.T
    # expand shape
    pbc_offsets = np.repeat(pbc_offsets.T[:, np.newaxis,:], natoms, axis=1)  # (27,natoms,3)
    cart_coords2 = np.repeat(cart_coords2[np.newaxis, :, :], num_cells, axis=0)  # (27,natoms,3)
    cart_coords1 = np.repeat(cart_coords1[np.newaxis, :, :], num_cells, axis=0)  # (27,natoms,3)
    cart_coords2 = cart_coords2 + pbc_offsets  # add offset of each cell

    atom_distance_vector = cart_coords1 - cart_coords2
    atom_distance_sqr = np.sum(atom_distance_vector**2, axis=-1)  # (27,natoms)
    min_atom_distance_sqr = np.min(atom_distance_sqr, axis=0)  # (natoms,)
    # print(min_atom_distance_sqr)
    min_atom_indices = np.argmin(atom_distance_sqr, axis=0)
    rmsd = np.sqrt(np.sum(min_atom_distance_sqr) / natoms)

    return rmsd


if __name__ == '__main__':
    cart_coords1 = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    cart_coords2 = [[0.0, 0.0, 1.9], [0.6, 0.6, 0.6]]
    lattice = [[1, 0, 0], [0, 1, 0], [0, 0, 2]]

    print(get_rmsd(cart_coords1, cart_coords2, lattice))
