import os
import subprocess
import warnings
import shutil
from itertools import product
from pathlib import Path

import click
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar, Potcar
from pymatgen.io.vasp.sets import MPRelaxSet
from joblib import Parallel, delayed
from tqdm import tqdm


def potcar2distmat(potcar: Potcar):
    rc = [p.RCORE for p in potcar]
    distmat = np.asarray([(i + j) * 0.529177 for i, j in product(rc, repeat=2)])
    distmat = distmat.reshape(len(rc), len(rc))
    return distmat


def vasp2inputdat(poscar, potcar, dist_ratio=0.7, popsize=1, p1=False):
    distmat = potcar2distmat(potcar) * dist_ratio
    ds = ""
    for line in distmat:
        ds += " ".join(map(str, line)) + "\n"

    inputdat = (
        f"SystemName = {''.join(poscar.site_symbols)}\n"
        f"NumberOfSpecies = {len(poscar.site_symbols)}\n"
        f"NameOfAtoms = {' '.join(poscar.site_symbols)}\n"
        f"NumberOfAtoms = {' '.join(map(str, poscar.natoms))}\n"
        "NumberOfFormula = 1 1\n"
        f"Volume = {poscar.structure.volume}\n"
        "@DistanceOfIon\n"
        f"{ds}"
        "@End\n"
        "Ialgo = 2\n"
        "PsoRatio = 0.6\n"
        f"PopSize = {popsize}\n"
        "ICode = 15\n"
        "NumberOfLbest = 4\n"
        "NumberOfLocalOptim = 3\n"
        "Command = sh submit.sh\n"
        "MaxStep = 5\n"
        "PickUp = F\n"
        "PickStep = 5\n"
        "Parallel = F\n"
        "Split = T\n"
    )
    if p1:
        inputdat += "SpeSpaceGroup = 1 1\n"
    return inputdat


def prepare_calypso_one(
    indir, fposcar, dist_ratio, popsize, calypsocmd, timeout, p1, label=""
):
    if len(label) > 0:
        outdir = indir.with_name(f"{indir.name}.calypso-{label}")
    else:
        outdir = indir.with_name(f"{indir.name}.calypso")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        poscar = Poscar.from_file(fposcar)
        potcar = Potcar.from_file(fposcar.with_name("POTCAR"))
    calypsodir = outdir.joinpath(fposcar.parent.relative_to(indir))
    calypsodir.mkdir(parents=True, exist_ok=True)
    with open(calypsodir.joinpath("input.dat"), 'w') as f:
        f.write(vasp2inputdat(poscar, potcar, dist_ratio, popsize, p1))
    for f in ["INCAR", "POTCAR", "KPOINTS"]:
        try:
            shutil.copy(fposcar.with_name(f), calypsodir)
        except FileNotFoundError:
            pass
            # logger.warning(f"{str(fposcar.with_name(f))} not found")
    # run calypso
    try:
        os.remove(calypsodir.joinpath("step"))
    except FileNotFoundError:
        pass
    with open(calypsodir.joinpath("caly.log"), "w") as calylog:
        try:
            proc = subprocess.run(
                calypsocmd,
                stdout=calylog,
                stderr=subprocess.STDOUT,
                cwd=calypsodir,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            pass
        else:
            # split POSCAR_* to subdir
            if proc.returncode != 0:
                print(f"Calling {calypsocmd} failed in {calypsodir}")
            else:
                for popi in range(1, popsize + 1):
                    calcdir = calypsodir.joinpath(f"calc/{popi}")
                    calcdir.mkdir(parents=True, exist_ok=True)
                    shutil.move(calypsodir / f"POSCAR_{popi}", calcdir / "POSCAR")
                    for f in ["INCAR", "POTCAR", "KPOINTS"]:
                        try:
                            shutil.copy(calypsodir / f, calcdir)
                        except FileNotFoundError:
                            pass
        finally:
            # clean dir
            for pyfile in calypsodir.glob("*.py"):
                os.remove(pyfile)


@click.command
@click.argument("indir", nargs=-1)
@click.option("-j", "--njobs", type=int, default=1, help="default 1")
@click.option("-r", "--dist_ratio", type=float, default=0.7, help="distance ratio on DistanceOfIon (default 0.7)")
@click.option("-p", "--popsize", type=int, default=1, help="Popsize (default 1)")
@click.option("-c", "--calypsocmd", type=str, default="calypso.x", help="Popsize (default 1)")
@click.option("-t", "--timeout", type=float, default=180, help="Each timeout seconds (default 180s)")
@click.option("--p1", is_flag=True, help="Only P1 if swich on else [1-230]")
@click.option("--label", default="", help="label")
def prepare_calypso(
    njobs, indir, dist_ratio, popsize, calypsocmd, timeout, p1, label
):
    for d in indir:
        d = Path(d)
        fposcarlist = list(d.rglob("POSCAR"))
        Parallel(njobs, backend="multiprocessing")(
            delayed(prepare_calypso_one)(
                d, fposcar, dist_ratio, popsize, calypsocmd, timeout, p1, label
            )
            for fposcar in tqdm(fposcarlist, ncols=79)
        )


if __name__ == "__main__":
    prepare_calypso()
