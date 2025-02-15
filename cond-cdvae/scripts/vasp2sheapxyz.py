# sheap xyz
# Lattice Properties name pressure energy spacegroup(symb) times_found=1
# SOPA-n*-l*-c*-g*             pbc
#         cutoff sigma

from itertools import chain
from pathlib import Path

import click
from ase.io import read, write
from ase.spacegroup import get_spacegroup
from dscribe.descriptors import SOAP
from joblib import Parallel, delayed
from tqdm import tqdm


def read_atoms(pdir: Path, fatoms: Path, soap: SOAP):
    pdir = pdir.resolve()
    fatoms = fatoms.resolve()
    name = str(fatoms.relative_to(pdir.parent.parent)).replace('/', '#')
    atoms = read(fatoms, format="vasp")
    soapkey = f"SOAP-n{soap._n_max}-l{soap._l_max}-c{soap._r_cut}-g{soap._sigma}"
    soapdesc = soap.create(atoms)
    atoms.info["name"] = name
    atoms.info["energy"] = 0.0
    atoms.info["spacegroup"] = get_spacegroup(atoms).symbol.replace(" ", "")
    atoms.info["times_found"] = 1
    atoms.info[soapkey] = " ".join(map(str, soapdesc))
    return atoms


def rglobvasp(fdir: Path):
    def key1(path: Path):
        stem = path.stem
        try:
            key = int(stem)
        except Exception:
            key = stem
        return key

    def key2(path: Path):
        name = path.parent.parent.parent.name
        try:
            key = int(name)
        except Exception:
            key = name
        return key

    part1 = list(
        sorted(
            chain(fdir.rglob("*.vasp")),
            key=key1,
        )
    )
    part2 = list(
        sorted(
            chain(fdir.rglob("*/calc/1/POSCAR")),
            key=key2,
        )
    )

    return part1 + part2


def vaspdir2soapxyz(njobs, vaspdirlist: list[Path], soapkwargs):
    soap = SOAP(**soapkwargs)
    for vaspdir in vaspdirlist:
        vaspflist = rglobvasp(vaspdir)
        atomslist = Parallel(njobs, "multiprocessing")(
            delayed(read_atoms)(vaspdir, f, soap)
            for f in tqdm(vaspflist, desc=str(vaspdir)[-20:])
        )
        write(vaspdir / "soap.xyz", atomslist, format="extxyz")


@click.command()
@click.argument('vaspdir', nargs=-1)
@click.option("-j", "--njobs", type=int, default=1)
@click.option("--n_max", default=5, type=int, help="n_max, default 5")
@click.option("--l_max", default=4, type=int, help="l_max, default 4")
@click.option("--r_cut", default=6, type=float, help="r_cut, default 6")
@click.option("--sigma", default=1.0, type=float, help="sigma, default 1.0")
@click.option("-E", "--species", multiple=True, required=True, help="element, multiple")
def main(njobs, vaspdir, n_max, l_max, r_cut, sigma, species):
    vaspdirlist = [Path(d) for d in vaspdir]
    soapkwargs = {
        "species": species,
        "n_max": n_max,
        "l_max": l_max,
        "r_cut": r_cut,
        "sigma": sigma,
        "periodic": True,
        "average": "inner",
    }
    vaspdir2soapxyz(njobs, vaspdirlist, soapkwargs)


if __name__ == "__main__":
    main()
