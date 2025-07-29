import click
import pandas as pd


@click.command
@click.option("--ds", help="batch generation dataset (feather)")
@click.option(
    "--mpds",
    default="/home/share/Data/MaterialsProject/mp-cif-230213.feather",
    help="Materials Project dataset (feather), "
    "default /home/share/Data/MaterialsProject/mp-cif-230213.feather",
)
@click.option(
    "--name",
    type=click.Choice(["mpexpe", "mptheo"]),
    default="mpexpe",
    help="part of dataset to write-out",
)
@click.option("--nperpart", default=600, type=int, help="number of part to split")
def split_dataset(ds, mpds, name, nperpart):
    ds = pd.read_feather(ds)
    mpds = pd.read_feather(mpds)
    theo_mpds = mpds[mpds.theoretical][["material_id", "formula"]]
    expe_mpds = mpds[~mpds.theoretical][["material_id", "formula"]]
    theo_ds = pd.merge(ds, theo_mpds, on="material_id", suffixes=("_x", ""))
    expe_ds = pd.merge(ds, expe_mpds, on="material_id", suffixes=("_x", ""))
    dsname = {"mptheo": theo_ds, "mpexpe": expe_ds}
    for ipart in range(len(dsname[name]) // nperpart + 1):
        dspart = dsname[name][ipart * nperpart : (ipart + 1) * nperpart]
        dspart.to_csv(
            f"{name}_p{ipart}.txt",
            index=False,
            sep=" ",
            columns=["material_id", "formula", "pressure"],
            header=False,
        )
        print(len(dspart))


if __name__ == "__main__":
    split_dataset()


# batch run generation
"""
python split_dataset.py \
    --ds /home/share/Data/Caly-MP/230617/v2_std_plus_DFTLi/test.feather \
    --nperpart 600
# >>> mpexpe_p{0..5}.txt
awk '{print "--label " $1 " --formula " $2 " --pressure " $3 }' mpexpe_p2.txt| \
    xargs -n 6 python ~/cond-cdvae/scripts/evaluate.py \
    --model_path `pwd` --batch_size 100 --num_batches_to_samples 2 --tasks gen
"""