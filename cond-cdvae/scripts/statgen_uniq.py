# 1. find the space group if no `spg.txt` (call find_spg.py)
# 2. grouping by the space group (with symprec=0.5) and find unique structures

from collections import defaultdict
from multiprocessing import Pool, RLock, freeze_support
from pathlib import Path

import click
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from statgen import to_format_table
from tqdm import tqdm

# from find_spg import get_spg
# from statgen import read_format_table


def get_matchers(levels):
    matcher_lo = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)  # loose
    matcher_md = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)  # midium
    matcher_st = StructureMatcher(ltol=0.1, stol=0.2, angle_tol=5)  # strict
    namelist = ["matcher_lo", "matcher_md", "matcher_st"]
    matcherlist = [matcher_lo, matcher_md, matcher_st]
    matchers = {namelist[level - 1]: matcherlist[level - 1] for level in levels}
    return matchers


def get_uniq_df(idx, gendir: Path, matchers):
    gen_list = sorted([int(f.stem) for f in gendir.glob("*.vasp")])

    genst_dict = {i: Structure.from_file(gendir / f"{i}.vasp") for i in gen_list}
    df = pd.DataFrame()
    for mat_name, matcher in matchers.items():
        pbar = tqdm(
            gen_list, ncols=160, desc=f"{gendir} {mat_name}", delay=1, position=idx
        )
        uniqid = defaultdict(list)
        for fi in pbar:
            formula = genst_dict[fi].composition.alphabetical_formula.replace(" ", "")
            df.loc[fi, "formula"] = formula
            for uid in uniqid[formula]:
                if matcher.fit(genst_dict[fi], genst_dict[uid]):
                    df.loc[fi, mat_name] = False
                    break
            else:
                df.loc[fi, mat_name] = True
                uniqid[formula].append(fi)
        pbar.close()

    # df = df.sort_values("formula")
    for mat_name, matcher in matchers.items():
        table_str = to_format_table(df[["formula", mat_name]])
        with open(gendir.joinpath(f"uniq.{mat_name[-2:]}.table"), "w") as f:
            f.write(table_str)

    return df


@click.command
@click.argument("gendirlist", nargs=-1)  # eval_gen*/gen
@click.option(
    "-l",
    "--levels",
    default=[1],
    multiple=True,
    type=int,
    help="number matcher, 1-loose, 2-mid, 3-strict",
)
@click.option("-j", "--njobs", default=-1, type=int, help="default: -1")
def filter_uniq(gendirlist, levels, njobs):
    assert njobs > 0, "njobs must be larger than 1"
    matchers = get_matchers(levels)
    gendirlist = [Path(d) for d in gendirlist if Path(d).is_dir()]

    freeze_support()
    pool = Pool(njobs, initializer=tqdm.set_lock, initargs=(RLock(),))
    for idx, gendir in enumerate(gendirlist):
        pool.apply_async(get_uniq_df, args=(idx, gendir, matchers))
    pool.close()
    pool.join()


if __name__ == '__main__':
    filter_uniq()
