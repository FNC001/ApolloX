import warnings
from pathlib import Path

import click
from joblib import Parallel, delayed
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import VaspInput
from pymatgen.io.vasp.sets import MPRelaxSet
from statgen import read_format_table
from tqdm import tqdm


def prepare_task(structure, relax_path, vaspargs):
    user_incar_settings = {
        'LREAL': False,
        'ISMEAR': 0,
        'NCORE': 4,
        'NSW': vaspargs["nsw"],
        'PSTRESS': vaspargs["pstress"],
        'ISYM': vaspargs["sym"],
    }
    if vaspargs["ediff"] is not None:
        user_incar_settings["EDIFF"] = vaspargs["ediff"]
    if vaspargs["ediffg"] is not None:
        user_incar_settings["EDIFFG"] = vaspargs["ediffg"]
    if vaspargs["kspacing"] is not None:
        user_incar_settings["KSPACING"] = vaspargs["kspacing"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp_set = MPRelaxSet(
            structure,
            user_incar_settings=user_incar_settings,
            user_potcar_settings={"W": "W_sv"},
            user_potcar_functional="PBE_54",
        )
        vasp = VaspInput(
            incar=mp_set.incar,
            kpoints=mp_set.kpoints,
            poscar=mp_set.poscar,
            potcar=mp_set.potcar,
        )
        vasp.write_input(relax_path)


def wrapped_prepare_task(indir, sf, vaspargs):
    runtype = ".scf" if vaspargs["nsw"] <= 1 else ".opt"
    relax_path = indir.with_name(f"{indir.name}{runtype}").joinpath(sf.stem)
    relax_path.mkdir(exist_ok=True, parents=True)

    structure = Structure.from_file(sf)
    prepare_task(structure, relax_path, vaspargs)


@click.command
@click.argument("indirlist", nargs=-1)
@click.option("-j", "--njobs", default=-1, type=int)
@click.option("-e", "--ediff", type=float, help="EDIFF, default None")
@click.option("-eg", "--ediffg", type=float, help="EDIFFG, default None")
@click.option("-s", "--nsw", default=0, help="NSW, mark scf if 0 else opt, default 0")
@click.option("-p", "--pstress", default=0, help="PSTRESS (kbar), default 0")
@click.option("-ks", "--kspacing", type=float,
              help="KSPACING, suggest 0.25, default None")
@click.option("--sym", type=int, default=0, help="ISYM, suggest 0/2, default 0")
def prepare_batch(indirlist, njobs, ediff, ediffg, nsw, pstress, kspacing, sym):
    vaspargs = {"ediff": ediff, "ediffg": ediffg, "nsw": nsw,
               "pstress": pstress, "kspacing": kspacing, "sym": sym}
    click.echo("You are using " + " ".join(f"{k}={v}" for k, v in vaspargs.items()))
    click.echo("Warning: W POTCAR is replaced by W_sv")
    for indir in indirlist:
        indir = Path(indir)
        flist = list(indir.glob("*.vasp"))
        Parallel(njobs, backend="multiprocessing")(
            delayed(wrapped_prepare_task)(indir, sf, vaspargs)
            for sf in tqdm(flist, ncols=120)
        )


if __name__ == '__main__':
    prepare_batch()
