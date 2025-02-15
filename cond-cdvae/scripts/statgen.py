import pickle
import subprocess
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure


def get_patched_mpds(mpds="/home/share/Data/MaterialsProject/mp-cif-230213.feather"):
    mpds = pd.read_feather(mpds)
    mpds.index = mpds.material_id
    return mpds


# [0, 20], [21, 40], [41, .]
def getnatomsgroup(mpds, material_id):
    natoms = mpds.loc[material_id, "nsites"]
    if natoms <= 20:
        return "0-20"
    elif natoms <= 40:
        return "21-40"
    else:
        return ">40"


def get_matchers():
    matcher_lo = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)  # loose
    matcher_md = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)  # midium
    matcher_st = StructureMatcher(ltol=0.1, stol=0.2, angle_tol=5)  # strict
    matchers = {
        "matcher_lo": matcher_lo,
        "matcher_md": matcher_md,
        "matcher_st": matcher_st,
    }
    return matchers


# match *.vasp with ground-truth structure(gtst) with each matcher in matchers
# return
#   matcher_lo matcher_md matcher_st
# 0        T/F        T/F        T/F
# 1        ...        ...        ...
def match_genstructure(
    gendir: Path,  # */gen
    gtst: Structure,
    matchers: dict[str, StructureMatcher],
    label,
):
    f_target = gendir.with_name(f"{label}.vasp")
    f_matchtable = gendir.with_name(f"match.{label}.table")
    gen_list = sorted([int(f.stem) for f in gendir.glob("*.vasp")])

    if f_matchtable.exists():
        df = pd.read_table(f_matchtable, sep=r"\s+", index_col="index")
        if len(df) >= len(gen_list):
            return df

    genst_dict = {i: Structure.from_file(gendir / f"{i}.vasp") for i in gen_list}

    data = {}
    for mat_name, matcher in matchers.items():
        fitdict = {i: matcher.fit(gtst, genst) for i, genst in genst_dict.items()}
        distdict = {
            i: (matcher.get_rms_dist(gtst, genst_dict[i]) if fit else (pd.NA, pd.NA))
            for i, fit in fitdict.items()
        }
        data[mat_name] = pd.Series(fitdict)
        data[f"{mat_name}_normrms"] = pd.Series({k: v[0] for k, v in distdict.items()})
        data[f"{mat_name}_maxrms"] = pd.Series({k: v[1] for k, v in distdict.items()})

    df = pd.DataFrame(data)

    # df = pd.DataFrame(
    #     {
    #         mat_name: pd.Series(
    #             {i: matcher.fit(gtst, genst) for i, genst in genst_dict.items()}
    #         )
    #         for mat_name, matcher in matchers.items()
    #     }
    # )

    gtst.to(str(f_target), fmt="poscar")
    table_str = to_format_table(df)
    with open(f_matchtable, "w") as f:
        f.write(table_str)

    return df


def load_dictpkl(picklefile):
    if Path(picklefile).exists():
        with open(picklefile, "rb") as f:
            d = pickle.load(f)
    else:
        d = {}
    return d


def to_format_table(df: pd.DataFrame, index_label="index"):
    csv_str = df.to_csv(
        None, sep=" ", float_format="%.6f", na_rep="NaN", index_label=index_label
    )
    fmt_proc = subprocess.Popen(
        ['column', '-t'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    table_str, _ = fmt_proc.communicate(csv_str)
    return table_str


def read_format_table(ftable):
    df = pd.read_table(ftable, sep=r"\s+", index_col=None)
    if "index" in df.columns:
        df = df.set_index("index")
    return df


def get_stress(pattern, *, inverse=False) -> dict:
    """get stress from OUTCAR

    Parameters
    ----------
    pattern : str
        string pattern like <pref>/*/<suff>/OUTCAR. Common <pref> and <suff> will be
        ignored, different parts will be treated as keys
    inverse : bool, default False
        find the last stress rather than the first one

    Returns
    -------
    stress_dict: dict
        {key: stress(GPa)}
    """
    # pattern: <pref>/*/<suff>/OUTCAR
    # eg. f"eval_gen_randtest-{p}G/std_5e-01.scf/*/OUTCAR"
    stress_dict = {}
    for foutcar in Path().glob(pattern):
        with open(foutcar, 'r') as f:
            outcar = f.readlines()
            if inverse:
                outcar = outcar[::-1]
            for line in outcar:
                if "external" in line:
                    linesp = line.split()
                    stress = (float(linesp[3]) + float(linesp[8])) / 10
                    break
            else:
                stress = np.nan
        stress_dict[str(foutcar)] = stress

    fsplit = [list(_) for _ in list(stress_dict.keys())]
    # >>> [[ABCxxxDEF], [ABCyyyDEF]]
    fsplit_T = [_ for _ in zip_longest(*fsplit, fillvalue=" ") if len(set(_)) != 1]
    # >>> [[A A], [B B], ...] -> [[x y], [x, y], ...]
    fsplit = ["".join(_).strip() for _ in zip(*fsplit_T)]
    # >>> [xxxDEF, yyyDEF]
    fsplit_inv = [_[::-1] for _ in fsplit]
    # >>> [[FEDxxx], [FEDyyy]]
    fsplit_inv_T = [
        _ for _ in zip_longest(*fsplit_inv, fillvalue=" ") if len(set(_)) != 1
    ]
    fsplit = ["".join(_[::-1]) for _ in zip(*fsplit_inv_T)]

    stress_dict = {k: v for k, v in zip(fsplit, stress_dict.values())}

    return stress_dict


def get_min_dist(atoms: Atoms) -> float:
    all_dist: np.ndarray = atoms.get_all_distances(mic=True)
    min_dist = all_dist[all_dist.nonzero()].min()
    return min_dist
