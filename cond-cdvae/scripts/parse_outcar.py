import pickle
from itertools import chain
from pathlib import Path

import click
import numpy as np
import pandas as pd
from ase.io import read
from joblib import Parallel, delayed
from statgen import to_format_table
from tqdm import tqdm

from statopt_rmsd import get_rmsd


def parse_one_outcar(foutcar: Path) -> pd.DataFrame:
    """Parse OUTCAR to pandas DataFrame

    .. code-block:: text
            formula energy volume PV extpressure converge cputime natoms nsites ...
       step
          0     ...

    set pd.NA if failed

    Parameters
    ----------
    foutcar : Path
        Path to OUTCAR

    Returns
    -------
    pd.DataFrame
        parsed properties of each step
    """
    foutcar = Path(foutcar)
    poscar_atoms = read(foutcar.with_name("POSCAR"), format="vasp")
    try:
        contcar_atoms = read(foutcar.with_name("CONTCAR"), format="vasp")
    except Exception:
        contcar_atoms = poscar_atoms
        # raise ValueError(f"read CONTCAR {foutcar.parent} error!")
    natoms = len(poscar_atoms)
    formula = poscar_atoms.get_chemical_formula("metal")
    rmsd = get_rmsd(
        contcar_atoms.get_positions(),
        poscar_atoms.get_positions(),
        poscar_atoms.get_cell()[:],
    )
    energylist = []  # eV
    Vlist = []
    PVlist = []  # eV
    extpres = []  # kbar
    total_drift_x = []
    total_drift_y = []
    total_drift_z = []
    converge = False
    cputime = pd.NA
    with open(foutcar, "r") as f:
        try:
            for line in f:
                if "energy  without" in line:
                    energylist.append(float(line.strip().split()[-1]))
                elif "P V=" in line:
                    PVlist.append(float(line.strip().split()[-1]))
                elif "volume of cell" in line:
                    Vlist.append(line.strip().split()[-1])
                elif "external pressure" in line:
                    extpres.append(line.strip().split()[3])
                elif "reached required" in line:
                    converge = True
                elif "CPU" in line:
                    cputime = float(line.strip().split()[-1])
                elif "total drift" in line:
                    total_drift = list(map(float, line.strip().split()[-3:]))
                    total_drift_x.append(total_drift[0])
                    total_drift_y.append(total_drift[1])
                    total_drift_z.append(total_drift[2])
        except Exception:
            print(f"May ERROR: {foutcar}")
    if len(energylist) == 0:
        energylist = [pd.NA]
        Vlist = [pd.NA]
        extpres = [pd.NA]
        total_drift_x = [pd.NA]
        total_drift_y = [pd.NA]
        total_drift_z = [pd.NA]
    else:
        Vlist = Vlist[1:]  # drop the duplicated first one
        Vlist = Vlist[: len(energylist)]
    if len(PVlist) == 0:
        PVlist = [0] * len(energylist)
    else:
        PVlist = PVlist[: len(energylist)]

    formula = [formula] * len(energylist)
    extpres = extpres[: len(energylist)]
    convergelist = [False] * len(energylist)
    convergelist[-1] = converge
    cputime = [cputime] * len(energylist)
    natoms = [natoms] * len(energylist)
    rmsd = [rmsd] * len(energylist)
    total_drift_x = total_drift_x[:len(energylist)]
    total_drift_y = total_drift_y[:len(energylist)]
    total_drift_z = total_drift_z[:len(energylist)]


    data = {
        "formula": formula,
        "energy": energylist,
        "total_drift_x": total_drift_x,
        "total_drift_y": total_drift_y,
        "total_drift_z": total_drift_z,
        "volume": Vlist,
        "PV": PVlist,
        "extpressure": extpres,
        "converge": convergelist,
        "cputime": cputime,
        "natoms": natoms,
        "nsites": natoms,
        "rmsd": rmsd,
    }
    # print({k: len(v) for k, v in data.items()})
    parsed_df = pd.DataFrame(data)
    parsed_df["enthalpy"] = parsed_df["energy"] + parsed_df["PV"]
    parsed_df["enthalpy_per_atom"] = parsed_df["enthalpy"] / parsed_df["natoms"]
    parsed_df.index.name = "step"
    return parsed_df


def stat_outcar_dfdict(dfdict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    serlist = []
    for fname, df in dfdict.items():
        total_drift0 = np.sqrt(
            (
                df[["total_drift_x", "total_drift_y", "total_drift_z"]]
                .iloc[0]
                .to_numpy()
                ** 2
            ).sum()
        )
        total_drift_1 = np.sqrt(
            (
                df[["total_drift_x", "total_drift_y", "total_drift_z"]]
                .iloc[-1]
                .to_numpy()
                ** 2
            ).sum()
        )
        H0 = df.at[0, "enthalpy"]
        H_1 = df.at[len(df) - 1, "enthalpy"]
        H_2 = df.at[len(df) - 2, "enthalpy"] if len(df) > 1 else np.nan

        ser = pd.Series(
            {
                "formula": df.at[0, "formula"],
                "converge": df.converge.iloc[-1],
                "enthalpy0": H0,
                "enthalpy-1": H_1,
                "decreased_enth": H0 - H_1,
                "final_d_enth": H_2 - H_1,
                "total_drift0": total_drift0,
                "total_drift-1": total_drift_1,
                "ion_steps": len(df),
                "natoms": df.at[0, "natoms"],
                "nsites": df.at[0, "nsites"],
                "cputime": df.at[0, "cputime"],
                "extpres0": df.at[0, "extpressure"],
                "extpres-1": df.at[len(df) - 1, "extpressure"],
                "rmsd": df.at[0, "rmsd"],
            },
            name=fname,
        )
        serlist.append(ser)
    stat_df = pd.DataFrame(serlist)
    stat_df["decreased_enth_per_atom"] = stat_df["decreased_enth"] / stat_df["natoms"]
    stat_df.index.name = "fname"
    return stat_df


def parse_outcar(indir, njobs, *args, **kwargs):
    indir = Path(indir)
    outcars = list(chain(indir.rglob("OUTCAR"), indir.rglob("*.OUTCAR")))
    if len(outcars) == 0:
        raise ValueError("No OUTCAR or *.OUTCAR found")
    parsed_dflist = Parallel(njobs, backend="multiprocessing")(
        delayed(parse_one_outcar)(foutcar) for foutcar in tqdm(outcars, ncols=79)
    )
    parsed_dfdict = {
        str(foutcar.relative_to(indir)): parsed_df
        for foutcar, parsed_df in zip(outcars, parsed_dflist)
    }
    stat_df = stat_outcar_dfdict(parsed_dfdict)
    print(stat_df)

    with open(indir.joinpath("parsed_outcar.pkl"), "wb") as f:
        parsed_df = pd.concat(parsed_dfdict, keys=parsed_dfdict.keys())
        parsed_df.index.names = ["fname", "step"]
        pickle.dump(parsed_df, f)
    with open(indir.joinpath("parsed_outcar.table"), "w") as f:
        f.write(to_format_table(stat_df))


@click.command()
@click.argument("indir", nargs=-1)
@click.option("-j", "--njobs", type=int, default=16)
def main(indir, njobs):
    for d in indir:
        parse_outcar(indir=d, njobs=njobs)


if __name__ == "__main__":
    main()
