# calcualte average coordinate number(CN) of a given element in a series of vasp files
# Algorithm 1: natural cutoff by ase
# Algorithm 2: Voronoi graph algorithm

import warnings
from itertools import chain
from pathlib import Path

import click
import numpy as np
import pandas as pd
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from joblib import Parallel, delayed
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.io.ase import AseAtomsAdaptor
from statgen import to_format_table
from tqdm import tqdm


def rglobvasp(fdir: Path):
    return list(
        chain(
            fdir.rglob("POSCAR"),
            fdir.rglob("POSCAR_*"),
            fdir.rglob("*.vasp"),
            fdir.rglob("poscar_*"),
            fdir.rglob("contcar_*"),
        )
    )


def avgcn_naturalcutoff(atoms, element):
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    cnlist = [
        len(nl.get_neighbors(i)[0])
        for i, atom in enumerate(atoms)
        if atom.symbol == element
    ]
    avgcn = np.mean(cnlist)
    return avgcn


def avgcn_voronoi(atoms, element):
    st = AseAtomsAdaptor.get_structure(atoms)
    CrystalNN = local_env.CrystalNN(
        # distance_cutoffs=None,
        x_diff_weight=-1,
        porous_adjustment=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        crystal_graph = StructureGraph.with_local_env_strategy(st, CrystalNN)
    edge_indices, to_jimages = [], []
    for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
        edge_indices.append([j, i])  # j as center
        to_jimages.append(to_jimage)
        edge_indices.append([i, j])  # i as center
        to_jimages.append(tuple(-tj for tj in to_jimage))
    cnlist = [
        sum(1 for j, _ in edge_indices if j == i)
        for i, atoms in enumerate(atoms)
        if atoms.symbol == element
    ]
    avgcn = np.mean(cnlist)
    return avgcn


def vasp2avgCN(fdir, njobs, elements):
    fdir = Path(fdir)
    vasplist = rglobvasp(fdir)
    for tag, func in zip(
        ["natcutoff", "voronoi"], [avgcn_naturalcutoff, avgcn_voronoi]
    ):
        avgcndict = {
            element: Parallel(njobs, "multiprocessing")(
                delayed(func)(read(fvasp, format="vasp"), element)
                for fvasp in tqdm(vasplist)
            )
            for element in elements
        }
        data = {"name": list(map(str, vasplist))}
        data.update(avgcndict)
        df = pd.DataFrame(data)
        with open(fdir / f"avgcn.{tag}.table", "w") as f:
            f.write(to_format_table(df))


@click.command()
@click.argument("dirlist", nargs=-1)
@click.option("-j", "--njobs", type=int)
@click.option("-E", "--elements", multiple=True)
def main(dirlist, njobs, elements):
    for fdir in dirlist:
        vasp2avgCN(fdir, njobs, elements)


if __name__ == "__main__":
    main()
