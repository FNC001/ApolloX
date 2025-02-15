import click
import pandas as pd


@click.command()
@click.argument("dataset")
@click.option("-f", "--formula", help="query formula")
@click.option("-i", "--material_id", help="query material_id")
@click.option("--spgno", type=int, help="spgno")
def search(dataset, formula=None, material_id=None, spgno=None):
    click.echo(dataset)
    ds = pd.read_feather(dataset)
    if formula is not None:
        if spgno is not None:
            click.echo(ds[(ds.formula == formula) & (ds.spgno == spgno)])
        else:
            click.echo(ds[ds.formula == formula])
    if material_id is not None:
        if spgno is not None:
            click.echo(ds[(ds.material_id == material_id) & (ds.spgno == spgno)])
        else:
            click.echo(ds[ds.material_id == material_id])


if __name__ == "__main__":
    search()
