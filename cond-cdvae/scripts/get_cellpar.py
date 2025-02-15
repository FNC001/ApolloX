from pathlib import Path
from itertools import chain

import click
import numpy as np
import pandas as pd
from ase.io import read
from joblib import Parallel, delayed

from statgen import to_format_table


def get_one_cellpar(fvasp):
    atoms = read(fvasp, format="vasp")
    return atoms.cell.cellpar()


@click.command()
@click.argument("fdir")
@click.option("-j", "--njobs", type=int, default=1)
def main(njobs, fdir):
    fdir = Path(fdir)
    fvasplist = list(chain(fdir.rglob("*.vasp"), fdir.rglob("POSCAR")))
    cellpar_list = Parallel(njobs, backend="multiprocessing")(
        delayed(get_one_cellpar)(fvasp) for fvasp in fvasplist
    )
    cellpar_list = np.array(cellpar_list)
    df = pd.DataFrame(
        cellpar_list, index=fvasplist, columns=["a", "b", "c", "alpha", "beta", "gamma"]
    )
    with open(fdir / "cellpar.table", "w") as f:
        f.write(to_format_table(df))

    df["a/b"] = df["a"] / df["b"]
    df["b/c"] = df["b"] / df["c"]
    df["c/a"] = df["c"] / df["a"]

    errdf = df[
        (df["a/b"] >= 5)
        | (df["a/b"] <= 0.2)
        | (df["b/c"] >= 5)
        | (df["b/c"] <= 0.2)
        | (df["c/a"] >= 5)
        | (df["c/a"] <= 0.2)
        | (df["alpha"] <= 20)
        | (df["alpha"] >= 160)
        | (df["beta"] <= 20)
        | (df["beta"] <= 20)
        | (df["gamma"] >= 160)
        | (df["gamma"] >= 160)
    ]

    print(errdf)


if __name__ == "__main__":
    main()
