# find the symmetry and standardlized cell of a given dir

import io
import subprocess
import sys
import shutil
from contextlib import redirect_stdout
from pathlib import Path

import click
import pandas as pd
import spglib
from ase import Atoms
from ase.io import read, write
from joblib import Parallel, delayed
from tqdm import tqdm


def atoms2cif(atoms):
    with io.BytesIO() as buffer, redirect_stdout(buffer):
        write('-', atoms, format='cif')
        cif = buffer.getvalue().decode()  # byte to string
    return cif


def atoms2vasp(atoms):
    with io.StringIO() as buffer, redirect_stdout(buffer):
        write('-', atoms, format='vasp')
        vasp = buffer.getvalue()  # byte to string
    return vasp


def get_spg_one(fname: Path, symprec_list, angle_tolerance=-1):
    atoms = read(fname)
    spg_dict = {
        "name": fname.name,
        "formula": atoms.get_chemical_formula("metal"),
    }
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    for symprec in symprec_list:
        symds = spglib.get_symmetry_dataset(cell, symprec, angle_tolerance)
        # ---- record
        if symds is not None:
            std_atoms = Atoms(
                symds["std_types"],
                cell=symds["std_lattice"],
                scaled_positions=symds["std_positions"],
            )
            spg_dict["{:.0e}".format(symprec)] = symds['number']
            spg_dict["{:.0e}".format(symprec) + "_symbol"] = symds['international']
            spg_dict["{:.0e}".format(symprec) + "_std_natoms"] = len(symds['std_types'])
            spg_dict["{:.0e}".format(symprec) + "_std_cif"] = atoms2cif(std_atoms)
            spg_dict["{:.0e}".format(symprec) + "_std_vasp"] = atoms2vasp(std_atoms)
        else:
            print(fname, symprec, "Cannot find symmetry", file=sys.stderr)
            spg_dict["{:.0e}".format(symprec)] = 0
            spg_dict["{:.0e}".format(symprec) + "_symbol"] = "-"
            spg_dict["{:.0e}".format(symprec) + "_std_natoms"] = 0
            spg_dict["{:.0e}".format(symprec) + "_std_cif"] = atoms2cif(atoms)
            spg_dict["{:.0e}".format(symprec) + "_std_vasp"] = atoms2vasp(atoms)
    return pd.Series(spg_dict)


def get_spg(fdir, symprec_list=(0.5, 0.1, 0.01)):
    flist = list(Path(fdir).glob("*.vasp"))
    symprec_list = sorted(symprec_list, reverse=True)
    ser_list = Parallel(-1, backend="multiprocessing")(
        delayed(get_spg_one)(f, symprec_list)
        for f in tqdm(flist, ncols=180, desc=f"{fdir}")
    )
    # ---- filter P1
    # !!!! write the original structure if not P1 under any symprec !!!!
    for fname, ser in zip(flist, ser_list):
        if any(ser["{:.0e}".format(symprec)] > 1 for symprec in symprec_list):
            sympart = fname.parent.parent.joinpath(f"sympart/{fname.parent.name}")
            sympart.mkdir(exist_ok=True, parents=True)
            shutil.copy(fname, sympart / fname.name)
            write(sympart / fname.with_suffix(".cif").name, read(fname))
    df = pd.DataFrame(ser_list)
    df = df.sort_values(by=list(map("{:.0e}".format, symprec_list)), ascending=False)
    return df


def to_format_csv(df: pd.DataFrame):
    df = df[[col for col in df if not col.endswith(("_std_cif", "_std_vasp"))]]
    csv_str = df.to_csv(None, sep=" ", index=False)
    fmt_proc = subprocess.Popen(
        ['column', '-t'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    output, _ = fmt_proc.communicate(csv_str)
    return output


def write_std_vasp(df: pd.DataFrame, indir):
    indir = Path(indir)
    cols = [col for col in df if col.endswith("_std_vasp")]
    prec_list = [col[:-9] for col in cols]
    for prec in prec_list:
        prec_dir = indir.with_name(f"{indir.name}.std_{prec}")
        prec_dir.mkdir(exist_ok=True)
        for _, ser in df.iterrows():
            with open(prec_dir / Path(ser['name']).name, "w") as fvasp:
                fvasp.write(ser[prec + "_std_vasp"])
            with open(prec_dir / f"{Path(ser['name']).stem}.cif", "w") as fcif:
                fcif.write(ser[prec + "_std_cif"])


@click.command(help="python find_spg.py eval_gen_*/gen [-s 0.5 -s 0.1 -s 0.001]")
@click.argument("indirs", nargs=-1)
@click.option(
    "-s",
    "--symprec",
    multiple=True,
    default=[0.5, 0.1, 0.01],
    help="symprec, can accept multiple time (not in one option)",
)
def cli_get_spg(indirs, symprec):
    spgdfdict = {}
    for indir in indirs:
        indir = Path(indir)
        df = get_spg(indir, symprec)
        csv = to_format_csv(df)
        with open(indir.with_name(f"{indir.name}.spg.txt"), 'w') as f:
            f.write(csv)
        write_std_vasp(df, indir)
        spgdfdict[indir.parent.name] = df


if __name__ == '__main__':
    cli_get_spg()
