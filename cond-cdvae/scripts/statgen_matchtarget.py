# match a dir of structures with one target
from pathlib import Path

import click
from pymatgen.core.structure import Structure
from statgen import get_matchers, match_genstructure


@click.command
@click.argument("gendir")  # eval_gen*/gen
@click.option("-t", "--target", required=True, help="target structure in vasp format")
@click.option("-p", "--picklefile", help="out txt, default: <target>.txt")
def matchtarget(gendir, target, picklefile):
    gendir = Path(gendir).resolve()
    target = Path(target).resolve()
    targetst = Structure.from_file(target)
    matchers = get_matchers()

    matchdf = match_genstructure(gendir, targetst, matchers, target.stem)
    return matchdf


if __name__ == "__main__":
    matchtarget()