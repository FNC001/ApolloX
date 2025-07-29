import shutil
from pathlib import Path

import click

from statgen import read_format_table, to_format_table


def filter_hydride(hydride_table, symprec, filter_nbondedHratio):
    hydride_table = Path(hydride_table)
    df = read_format_table(hydride_table)
    sympreckey = "{:.0e}".format(symprec)
    if sympreckey not in df:
        raise KeyError(f"symprec {sympreckey} not in given table")
    df = df[df[sympreckey] > 2]
    df = df[["name", sympreckey, "formula_hill", "nsites", "mindist", "nH", "nbondedH", "nonMcageH"]]  # fmt: skip
    df.insert(df.columns.get_loc("nbondedH") + 1, "nbondedHratio", df.nbondedH / df.nH)
    df = df.sort_values(["nbondedHratio", sympreckey], ascending=[True, False])
    # print(to_format_table(df))
    eps = 1e-5
    print("nbondedHratio <= 1e-5 : ", len(df[df.nbondedHratio <= eps]))
    print("nbondedHratio <= 1e-1 : ", len(df[df.nbondedHratio <= 0.1 + eps]))
    print("nbondedHratio <= 2e-1 : ", len(df[df.nbondedHratio <= 0.2 + eps]))
    print("nbondedHratio <= 3e-1 : ", len(df[df.nbondedHratio <= 0.3 + eps]))
    with open(hydride_table.with_suffix(".filter.table"), "w") as f:
        f.write(to_format_table(df))

    # write filtered file
    filter_dir = hydride_table.with_name(f"{hydride_table.name[:-14]}.filter")
    shutil.rmtree(filter_dir, ignore_errors=True)
    filter_dir.mkdir()
    for idx, ser in df[df.nbondedHratio <= filter_nbondedHratio + eps].iterrows():
        shutil.copy(hydride_table.parent.joinpath(ser["name"]), filter_dir)


@click.command()
@click.argument("hydride_table", nargs=-1)
@click.option("-s", "--symprec", type=float, default=0.1, help="symprec to use, default 0.1")  # fmt: skip
@click.option("--filter-nbondedHratio", type=float, default=0.3, help="filter nbondedHratio, default 0.3")  # fmt: skip
def main(hydride_table, symprec, filter_nbondedhratio):
    for ftable in hydride_table:
        print("filtering ", ftable)
        filter_hydride(ftable, symprec, filter_nbondedhratio)


if __name__ == "__main__":
    main()
