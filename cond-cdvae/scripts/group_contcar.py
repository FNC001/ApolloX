import shutil
from itertools import chain
from pathlib import Path

import click


@click.command(help="Group all CONTCAR to <contcars> and <contcars.txt>")
@click.argument("dirlist", nargs=-1)
@click.option("--targetdir", default="contcars", help="target output, default contcars")
def main(dirlist, targetdir):
    targetdir = Path("contcars")
    shutil.rmtree(targetdir, ignore_errors=True)
    targetdir.mkdir(parents=True)

    with open(targetdir.joinpath("idx.txt"), "w") as fidx:
        for idx, fcontcar in enumerate(
            chain.from_iterable(Path(d).rglob("CONTCAR") for d in dirlist)
        ):
            shutil.copy(fcontcar, targetdir.joinpath(f"{idx}.vasp"))
            fidx.write(f"{idx}.vasp     {fcontcar}\n")


if __name__ == "__main__":
    main()
