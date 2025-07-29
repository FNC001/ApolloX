import io
from contextlib import redirect_stdout
from pathlib import Path

import click
import pandas as pd
from ase import Atoms
from ase.io import read, write
from spglib import standardize_cell


def series2atoms(series: pd.Series):
    atoms = read(io.StringIO(series.cif), format="cif")
    return atoms


def atoms2cifstring(atoms):
    with io.BytesIO() as buffer, redirect_stdout(buffer):
        write('-', atoms, format='cif')
        cif = buffer.getvalue().decode()  # byte to string
    return cif


def standardize_feather(df, to_primitive=False):
    serlist = []
    for idx, ser in df.iterrows():
        atoms = series2atoms(ser)
        bulk = (
            atoms.get_cell(),
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers(),
        )
        stdcell = standardize_cell(bulk, to_primitive)
        if stdcell is not None:
            lattice, scaled_positions, numbers = stdcell
            atoms = Atoms(
                numbers, cell=lattice, scaled_positions=scaled_positions, pbc=True
            )
            ser["formula"] = atoms.get_chemical_formula("metal")
            ser["cif"] = atoms2cifstring(atoms)
            ser["natoms"] = len(atoms)
            ser["nsites"] = len(atoms)
        serlist.append(ser)
    df = pd.DataFrame(serlist)
    return df


@click.command()
@click.argument("featherfile")
@click.option(
    "-p",
    "--primitive/--no-primitive",
    default=False,
    help="to primitive cell or not, default False",
)
def main(featherfile, primitive):
    featherfile = Path(featherfile)
    df = pd.read_feather(featherfile)
    df = standardize_feather(df, primitive)
    celltype = "pcell" if primitive else "ucell"
    ofile = featherfile.with_name(".".join([featherfile.stem, celltype, "feather"]))
    click.echo(f"saving to {ofile}")
    df.to_feather(ofile)


if __name__ == "__main__":
    main()
