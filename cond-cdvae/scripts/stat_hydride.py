import warnings
from collections import defaultdict
from itertools import chain
from pathlib import Path

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar
from tqdm import tqdm

from find_spg import get_spg_one
from statgen import to_format_table


def rglobvasp(fdir: Path):
    fdir = Path(fdir)
    return list(
        chain(
            fdir.rglob("POSCAR"),
            fdir.rglob("POSCAR_*"),
            fdir.rglob("*.vasp"),
            fdir.rglob("poscar_*"),
            fdir.rglob("contcar_*"),
        )
    )


class HydrideAnalyzer:
    def __init__(self, fvasp, max_Hbond=1.0, symprec_list=[0.1]):
        self.CrystalNN = local_env.CrystalNN(
            x_diff_weight=-1,
            porous_adjustment=False,
        )
        self.fvasp = fvasp
        self.structure = Poscar.from_file(fvasp).structure
        if self.graph is None:
            print(f"{self.fvasp} does not have valid crystal graph, set to None and skip initialization")  # fmt: skip
            return
        self.max_Hbond = max_Hbond
        self.symprec_list = symprec_list
        self.natoms = self.nsites = len(self.structure)

        self.mindist = np.min(self.distmat)
        self.nH = len(self.Hindices)
        self.nbondedH = self.get_nbondedH()
        self.nMcageH = len(self.McageH)
        self.nonMcageH = self.nH - self.nMcageH

    @property
    def series(self):
        info = {}
        if self.graph is None:
            return pd.Series(info)
        info.update(self.spgdict)
        info.update(
            {key: getattr(self, key)
            for key in ["formula_hill", "nsites", "mindist", "nH", "nbondedH", "nonMcageH",]}
        )  # fmt: skip
        info.update(
            {f"min_H-M{i}": np.min(self.dist_HM[k], initial=np.inf)
            for i, k in enumerate(sorted(self.dist_HM.keys()))}
        )  # fmt: skip
        info.update(
            {f"avg_H-M{i}": np.nan if len(self.dist_HM[k]) == 0 else np.mean(self.dist_HM[k])
            for i, k in enumerate(sorted(self.dist_HM.keys()))}
        )  # fmt: skip
        info.update(
            {f"std_H-M{i}": np.nan if len(self.dist_HM[k]) == 0 else np.std(self.dist_HM[k])
            for i, k in enumerate(sorted(self.dist_HM.keys()))}
        )  # fmt: skip
        return pd.Series(info)

    @property
    def spgdict(self):
        symkeys = ["{:.0e}".format(symprec) for symprec in self.symprec_list]
        if getattr(self, "_spgdict", None) is None:
            spg_ser = get_spg_one(self.fvasp, self.symprec_list)
            self._spgdict = spg_ser[symkeys].to_dict()
        return self._spgdict

    @property
    def Hindices(self):
        """indices of H"""
        if getattr(self, "_Hindices", None) is None:
            self._Hindices = [
                i for i, n in enumerate(self.structure.atomic_numbers) if n == 1
            ]
        return self._Hindices

    @property
    def Mindices(self):
        """indices of non-H"""
        if getattr(self, "_Mindices", None) is None:
            self._Mindices = [
                i for i, n in enumerate(self.structure.atomic_numbers) if n != 1
            ]
        return self._Mindices

    def get_nbondedH(self):
        Hdistmat = np.take(self.distmat, self._Hindices, 0)
        return np.any(Hdistmat <= self.max_Hbond, axis=1).sum()

    @property
    def distmat(self):
        if getattr(self, "_distmat", None) is None:
            self._distmat = self.structure.distance_matrix
            np.fill_diagonal(self._distmat, min(self.structure.lattice.abc))
        return self._distmat

    @property
    def formula_hill(self):
        if getattr(self, "_formula_hill", None) is None:
            atoms = AseAtomsAdaptor.get_atoms(self.structure)
            self._formula_hill = atoms.get_chemical_formula(mode="hill")
        return self._formula_hill

    @property
    def graph(self):
        """[[i, j, jix, jiy, jiz], ...]"""
        if getattr(self, "_graph", None) is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                graph = StructureGraph.with_local_env_strategy(
                    self.structure, self.CrystalNN
                ).graph
            edges = np.array(
                [
                    [i, j] + list(to_jimage)
                    for i, j, to_jimage in graph.edges(data='to_jimage')
                ],
                dtype=np.int64,
            )
            # reverse edge direction and flip to_jimage
            if len(edges) == 0:
                self._graph = None
                return self._graph
            else:
                rev_edges = edges[:, [1, 0, 2, 3, 4]] * np.array([[1, 1, -1, -1, -1]])
            self._graph = np.concatenate([edges, rev_edges])
        return self._graph

    @property
    def Mcages(self):
        """dict of each M's cage vertexes, {Mi: [[j, ...], ...]}"""
        if getattr(self, "_Mcages", None) is None:
            self._Mcages = {
                Mi: [j for i, *j in self.graph if i == Mi] for Mi in self.Mindices
            }
        return self._Mcages

    @property
    def McageH(self):
        """All H indices surrounding M"""
        if getattr(self, "_McageH", None) is None:
            cage_vertexes = [
                vertex[0] for Mcage in self.Mcages.values() for vertex in Mcage
            ]
            self._McageH = list(set(self.Hindices) & set(cage_vertexes))
        return self._McageH

    @property
    def dist_HM(self):
        """dict of distance list of H to each M"""
        if getattr(self, "_d_HM", None) is None:
            self._d_HM = defaultdict(list)
            for Mi, Mcage in self.Mcages.items():
                Hilist = [j for j, *_ in Mcage if j in self.Hindices]
                d_MH_list = [self.distmat[Mi, Hi] for Hi in Hilist]
                self._d_HM[f"d_H-{self.structure[Mi].species_string}"] += d_MH_list
        return self._d_HM


def wrapped_analysis(fvasp, max_Hbond, symprec_list):
    analyzer = HydrideAnalyzer(fvasp, max_Hbond, symprec_list)
    return analyzer.series


def analysis(njobs, vaspdirlist: list[Path], max_Hbond, symprec_list):
    for vaspdir in vaspdirlist:
        vaspflist = rglobvasp(vaspdir)
        # structures = [Poscar.from_file(fvasp).structure for fvasp in vaspflist]
        series_list = Parallel(njobs, "multiprocessing")(
            delayed(wrapped_analysis)(fvasp, max_Hbond, symprec_list)
            for fvasp in tqdm(
                vaspflist, desc=str(vaspdir)[-20:], ncols=120, mininterval=1
            )
        )
        namelist = [str(fvasp.relative_to(vaspdir.parent)) for fvasp in vaspflist]
        df = pd.DataFrame(series_list)
        df.insert(0, "name", namelist)
        sort_keys = ["{:.0e}".format(symp) for symp in symprec_list] + ["nbondedH", "nonMcageH",]  # fmt: skip
        ascending = [False] * len(symprec_list)                      + [True,       True]  # fmt: skip
        df = df.sort_values(sort_keys, ascending=ascending)
        with open(vaspdir.parent.joinpath(f"{vaspdir.name}.hydride.table"), "w") as f:
            f.write(to_format_table(df))
        print(df)


@click.command()
@click.argument('vaspdir', nargs=-1)
@click.option("-j", "--njobs", type=int, default=1)
@click.option("--max_Hbond", type=float, default=1.0, help="max H bond (default 1.0)")
@click.option("-s", "--symprec", multiple=True, default=[0.1], help="symprec, can accept multiple time (not in one option)")  # fmt: skip
def main(vaspdir, njobs, max_hbond, symprec):
    vaspdir = [Path(d) for d in vaspdir]
    analysis(njobs, vaspdir, max_hbond, symprec)


if __name__ == "__main__":
    main()
