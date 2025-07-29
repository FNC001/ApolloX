from pathlib import Path

import click
import pandas as pd
from ase.io import read
from statgen import get_min_dist, to_format_table
from tqdm import tqdm
from joblib import Parallel, delayed


def parse_one(fname):
    atoms = read(fname)
    min_dist = get_min_dist(atoms)
    ser = pd.Series(
        {
            'name': Path(fname).name,
            'formula': atoms.get_chemical_formula('metal'),
            'min_dist': min_dist,
            'volume_per_atom': atoms.get_volume() / len(atoms),
        }
    )
    return ser


@click.command
@click.argument("fdir")
@click.option("-j", "--njobs", default=64, type=int, help="parallel jobs")
def main(fdir, njobs):
    flist = list(Path(fdir).glob('*.vasp'))
    serlist = Parallel(njobs)(
        delayed(parse_one)(fname) for fname in tqdm(flist, ncols=119)
    )
    df = pd.DataFrame(serlist)
    df = df.sort_values(by='min_dist', ascending=False)
    with open(Path(fdir).with_name("mindist.table"), "w") as f:
        f.write(to_format_table(df))


if __name__ == "__main__":
    main()
