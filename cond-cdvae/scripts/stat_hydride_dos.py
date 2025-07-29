# vaspkit needs to be run first
# ISPIN should be 2

from pathlib import Path

import click
import numpy as np
import pandas as pd
from ase.io import read
from scipy import interpolate

from statgen import read_format_table, to_format_table


def parse_vaspkit_dos(vaspdir):
    atoms = read(vaspdir / "../POSCAR")
    volume = atoms.get_volume()
    fpdoslist = list(vaspdir.glob("PDOS_*.dat"))
    if len(fpdoslist) == 0:
        fermidos = {elem: np.nan for elem in atoms.symbols.formula.count().keys()}
    else:
        elemlist = list(set(fpdos.stem.split("_")[1] for fpdos in fpdoslist))
        fermidos = {elem: 0 for elem in elemlist}
        for elem in elemlist:
            for spin in ["UP", "DW"]:
                pdos = pd.read_csv(vaspdir / f"PDOS_{elem}_{spin}.dat", sep=r"\s+")
                interp_f = interpolate.interp1d(x=pdos["#Energy"], y=pdos["tot"])
                fermidos[elem] += abs(interp_f(0.0))
            fermidos[elem] /= volume
    return fermidos


@click.command()
@click.argument("dosparentdir")
@click.option(
    "-t",
    "--hydride_table",
    required=True,
    help="hydride info, usually `*.hydride.filter.table`",
)
def main(dosparentdir, hydride_table):
    dosparentdir = Path(dosparentdir)
    fermidos_serlist = []
    for doscar in dosparentdir.glob("*/dos/DOSCAR"):
        vaspdir = doscar.parent
        fermidos = parse_vaspkit_dos(vaspdir)
        fermidos_serlist.append(
            pd.Series(
                {
                    "idname": doscar.parent.parent.name,
                    "fermidos_H": fermidos["H"],
                    "fermidos_tot": sum(fermidos.values()),
                }
            )
        )
    df = pd.DataFrame(fermidos_serlist)

    f_hydride_table = Path(hydride_table)
    hydride_table = read_format_table(f_hydride_table)
    hydride_table["idname"] = hydride_table["name"].apply(lambda f: Path(f).stem)

    df = pd.merge(hydride_table, df, how="right", on="idname")
    df = df.drop("idname", axis=1)
    df["fermidos_Hratio"] = df["fermidos_H"] / df["fermidos_tot"]
    sympreckey = df.columns[1]
    df = df.sort_values(
        # ["nbondedHratio", sympreckey, "fermidos_Hratio", "fermidos_H"],
        # ascending=[True, False, False, False],
        [sympreckey, "fermidos_Hratio", "fermidos_H"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    print(df)

    with open(dosparentdir.with_name(f"{dosparentdir.name}.dos.table"), "w") as f:
        f.write(to_format_table(df))


if __name__ == "__main__":
    main()
