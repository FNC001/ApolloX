import time
from pathlib import Path
from types import SimpleNamespace

import click
import torch
from eval_utils import load_custom_dataset, load_model
from evaluate import reconstruction
from torch_geometric.loader import DataLoader


def custom_reconstruction(
    loader,
    model,
    ld_kwargs,
    num_evals=1,
    down_sample_traj_step=1,
):
    return reconstruction(loader, model, ld_kwargs, num_evals, down_sample_traj_step)


@click.command
@click.option("-m", "--model_path")
@click.option("-d", "--data_path", help="data table path, eg. *.feather")
@click.option("-b", "--batch_size", type=int, default=1)
@click.option("--label", default="", help="output to eval_recon_{label}.pt")
def main(model_path, data_path, batch_size, label=""):
    model_path = str(Path(model_path).resolve())
    data_path = str(Path(data_path).resolve())
    ld_kwargs = SimpleNamespace(
        n_step_each=100,
        step_lr=1e-4,
        min_sigma=0,
        save_traj=False,
        disable_bar=False,
    )
    recon_out_name = f'eval_recon_{label}.pt'
    # ========= to save ===============
    eval_setting = vars(ld_kwargs)
    eval_setting["model_path"] = model_path
    eval_setting["data_path"] = data_path
    eval_setting["label"] = label
    # ==================================

    model, _, cfg = load_model(model_path)
    custom_dataset = load_custom_dataset(model_path, data_path)

    loader = DataLoader(custom_dataset, batch_size)
    start_time = time.time()
    (
        frac_coords,
        num_atoms,
        atom_types,
        lengths,
        angles,
        all_frac_coords_stack,
        all_atom_types_stack,
        input_data_batch,
    ) = custom_reconstruction(loader, model, ld_kwargs)

    torch.save(
        {
            'eval_setting': eval_setting,
            'input_data_batch': input_data_batch,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time,
        },
        Path(model_path) / recon_out_name,
    )


if __name__ == "__main__":
    main()
