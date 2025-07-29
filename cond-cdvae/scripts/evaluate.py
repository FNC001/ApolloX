import argparse
import pickle
import random
import re
import time
from collections import Counter
from itertools import chain, zip_longest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from eval_utils import composition2atom_types, load_model
from pymatgen.core.composition import Composition, Element
from torch.optim import Adam
from torch_geometric.data import Batch
from tqdm import tqdm


def reconstruction(
    loader,
    model,
    ld_kwargs,
    num_evals,
    down_sample_traj_step=1,
):
    """
    reconstruct the crystals in <loader>.
    """
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    input_data_list = []

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        print(f'batch {idx} in {len(loader)}')
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        # only sample one z, multiple evals for stoichaticity in langevin dynamics
        conditions = model.build_conditions(batch)
        c_dict = model.multiemb(conditions)
        cond_vec = model.agg_c(z=None, c=c_dict)
        _, _, z = model.encode(batch, cond_vec)
        # conditional z
        cond_z = model.zgivenc(z, c_dict)  # z (B, *)

        for eval_idx in range(num_evals):
            gt_num_atoms = batch.num_atoms
            gt_atom_types = batch.atom_types
            outputs = model.langevin_dynamics(
                cond_z, ld_kwargs, gt_num_atoms, gt_atom_types
            )

            # collect sampled crystals in this batch.
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lengths.append(outputs['lengths'].detach().cpu())
            batch_angles.append(outputs['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    outputs['all_frac_coords'][::down_sample_traj_step].detach().cpu()
                )
                batch_all_atom_types.append(
                    outputs['all_atom_types'][::down_sample_traj_step].detach().cpu()
                )
        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(torch.stack(batch_all_atom_types, dim=0))
        # Save the ground truth structure
        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords,
        num_atoms,
        atom_types,
        lengths,
        angles,
        all_frac_coords_stack,
        all_atom_types_stack,
        input_data_batch,
    )


periodic_table = (
    'H', 'He',
    'Li', 'Be', 'B',  'C',  'N', 'O',  'F',  'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S',  'Cl', 'Ar',
    'K',  'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe',
    'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra',
    'Ac','Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue'
)  # fmt: skip
M = (
    # 'H', 'He',
    'Li', 'Be', 'B',  'C',  'N', 'O',  'F',  'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S',  'Cl', 'Ar',
    'K',  'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe',
    'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', # 'Po', 'At', 'Rn',
    # 'Fr', 'Ra',
    # 'Ac','Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    # 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue'
)  # fmt: skip
range_pat = re.compile(r"\d+\-\d+")
# `hydride` match HxMyMz
hydride_base_pat = re.compile(r"\((?P<hydride>H\d+([A-Za-z]+\d+)+)\)\d+")


def replace_hydride_formula(string):
    """replace formula string into hydride formula string

    Replace every 'M' to one of M
    """
    formula = ""
    for i in string:
        if i == 'M':
            formula += random.choice(M)
        else:
            formula += i
    return formula


def sample_range(string):
    """find each range and sample from it

    Replace m-n to k, m<=k<=n
    """
    nrange = range_pat.findall(string)
    if len(nrange) == 0:
        result_string = string
    else:
        parts = re.split(range_pat, string)
        result_string = ""
        for irange, part in zip_longest(nrange, parts):
            if irange is None:
                ni = ""
            else:
                start, stop = tuple(map(int, irange.split("-")))
                if start == stop == 0:
                    raise ValueError("range should not be 0-0")
                ni = str(random.randint(start, stop))
            result_string += f"{part}{ni}"
    return result_string


def sample_formula_range(string: str, *, hydride=False):
    """sample a formula from range

    Parameters
    ----------
    string : str
        formula string with range (e.g. H4-8O2-4, (H2O)2-4, (H2O)1-2(CaO)3-4 ...)
    hydride : bool = False
        hydride mode, string must be like '(HxMyMzLan)m'. 'M' will be replace to M,
        other element like La is optional. (xyznm) should be a single int or int range.

    Returns
    -------
    str
        formula string without range
    """
    result_string = sample_range(string)
    if hydride:
        hydride_base = re.match(hydride_base_pat, result_string)  # in parenthesis part
        if hydride_base is None:
            raise ValueError("hydride format not match")
        else:
            hydride_base = hydride_base.group("hydride")
        nlist = list(map(int, re.findall(r"\d+", hydride_base)))
        # make sure at least one M
        while sum(nlist[1:]) == 0:
            result_string = sample_range(string)
            hydride_base = re.match(hydride_base_pat, result_string).group("hydride")
            nlist = list(map(int, re.findall(r"\d+", hydride_base)))
        result_string = replace_hydride_formula(result_string)
    return result_string


def generation(
    model,
    ld_kwargs,
    num_batches_to_sample,
    num_samples_per_z,
    batch_size=512,
    down_sample_traj_step=1,
    formula=None,
    hydride=False,
    train_data=None,
    **norm_target_props,
):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    formulabak = formula  # a backup
    for batch_idx in range(num_batches_to_sample):
        print(f"generate on batch {batch_idx}/{num_batches_to_sample}")
        # init args for this batch
        if not (formulabak is None) ^ (train_data is None):
            raise Exception("formula and train_data should only specify one")
        elif formulabak is not None:
            formula_list = [
                sample_formula_range(formulabak, hydride=hydride)
                for _ in range(batch_size)
            ]
            sampled_num_atoms = []
            sampled_atom_types = []
            for formula in formula_list:
                comp = Composition(formula)
                specified_atom_types = torch.tensor(composition2atom_types(comp))
                sampled_num_atoms.append(len(specified_atom_types))
                sampled_atom_types.append(specified_atom_types)
            sampled_num_atoms = torch.tensor(sampled_num_atoms, device=model.device)
            sampled_atom_types = torch.hstack(sampled_atom_types)
            sampled_atom_types = sampled_atom_types.to(model.device)
        elif train_data is not None:  # load cached data
            cached_data = pickle.load(open(train_data, 'rb'))

            comp_counts = dict(
                Counter(
                    Composition(dict(Counter(sample["graph_arrays"][1])))
                    for sample in cached_data  # [atomic_number list]
                )
            )
            sampled_comps = random.choices(
                population=list(comp_counts.keys()),
                weights=list(comp_counts.values()),
                k=batch_size,
            )
            sampled_atom_types = list(
                chain.from_iterable(
                    composition2atom_types(comp) for comp in sampled_comps
                )
            )
            sampled_num_atoms = [comp.num_atoms for comp in sampled_comps]
            sampled_atom_types = torch.tensor(sampled_atom_types, device=model.device)
            sampled_num_atoms = torch.tensor(sampled_num_atoms, device=model.device)
        # return `sampled_atom_types` and `sampled_num_atoms`
        conditions = {}
        for k, v in norm_target_props.items():
            val = [v] * batch_size
            val =torch.tensor(val, device=model.device, dtype=torch.get_default_dtype())
            if val.dim() == 1:
                val = val.unsqueeze(1)
            conditions[k] = val
        conditions['composition'] = (sampled_atom_types, sampled_num_atoms)
        # return conditions dict

        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        # z & cond z
        c_dict = model.multiemb(conditions)
        z = torch.randn(batch_size, model.hparams.latent_dim, device=model.device)
        # conditional z
        cond_z = model.zgivenc(z, c_dict)  # z (B, *)

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics(
                cond_z, ld_kwargs, sampled_num_atoms, sampled_atom_types
            )

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lengths.append(samples['lengths'].detach().cpu())
            batch_angles.append(samples['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples['all_frac_coords'][::down_sample_traj_step].detach().cpu()
                )
                batch_all_atom_types.append(
                    samples['all_atom_types'][::down_sample_traj_step].detach().cpu()
                )

        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (
        frac_coords,
        num_atoms,
        atom_types,
        lengths,
        angles,
        all_frac_coords_stack,
        all_atom_types_stack,
    )


def target_generation(
    model,
    ld_kwargs,
    num_samples_per_z,
    num_batches_to_samples,
    target_file: str,
    batch_size=512,
    down_sample_traj_step=1,
):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    material_id_list = []

    target_file = Path(target_file)
    if not target_file.is_absolute():
        raise FileExistsError("target_file must be absolute")
    elif not target_file.exists():
        raise FileNotFoundError(f"{target_file}")
    elif target_file.suffix == ".csv":
        target_df = pd.read_csv(target_file)
    else:
        raise FileExistsError("target_file must be a csv")

    for key in ["formula", "pressure"]:
        if key not in target_df.columns:
            raise ValueError(f"{key} must be in target csv file")

    groups = pd.Index(range(len(target_df))) // batch_size
    nbatches = np.ceil(len(target_df) / batch_size)
    for repeat_idx in range(num_batches_to_samples):
        for batch_idx, batch_df in target_df.groupby(groups):
            print(f"generate on {repeat_idx=}/{num_batches_to_samples} , {batch_idx=}/{nbatches}")
            real_batch_size = len(batch_df)
            target_dict = batch_df.to_dict("list")

            batch_material_id = target_dict.pop("material_id", None)
            if batch_material_id is not None:
                batch_material_id = batch_material_id * num_samples_per_z
                material_id_list += batch_material_id

            conditions = {}
            for key, val_list in target_dict.items():
                if key == "formula":
                    sampled_num_atoms = [None] * len(val_list)
                    sampled_atom_types = [None] * len(val_list)
                    for j, formula in enumerate(val_list):
                        comp = Composition(formula)
                        specified_atom_types = torch.tensor(composition2atom_types(comp))
                        sampled_num_atoms[j] = len(specified_atom_types)
                        sampled_atom_types[j] = specified_atom_types
                    sampled_num_atoms = torch.tensor(sampled_num_atoms, device=model.device)
                    sampled_atom_types = torch.hstack(sampled_atom_types)
                    sampled_atom_types = sampled_atom_types.to(model.device)
                    conditions["composition"] = (sampled_atom_types, sampled_num_atoms)
                    print("Add key composition")
                else:
                    try:
                        cond_value = torch.tensor(
                            val_list,
                            device=model.device,
                            dtype=torch.get_default_dtype()
                        ).view(-1, 1)
                    except Exception:
                        print(f"Key {key} failed to build target, skip")
                    else:
                        conditions[key] = cond_value
                        print(f"Add key {key}")

            batch_all_frac_coords = []
            batch_all_atom_types = []
            batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
            batch_lengths, batch_angles = [], []

            # z & cond z
            c_dict = model.multiemb(conditions)
            z = torch.randn(real_batch_size, model.hparams.latent_dim, device=model.device)
            # conditional z
            cond_z = model.zgivenc(z, c_dict)  # z (B, *)

            for sample_idx in range(num_samples_per_z):
                samples = model.langevin_dynamics(
                    cond_z, ld_kwargs, sampled_num_atoms, sampled_atom_types
                )

                # collect sampled crystals in this batch.
                batch_frac_coords.append(samples['frac_coords'].detach().cpu())
                batch_num_atoms.append(samples['num_atoms'].detach().cpu())
                batch_atom_types.append(samples['atom_types'].detach().cpu())
                batch_lengths.append(samples['lengths'].detach().cpu())
                batch_angles.append(samples['angles'].detach().cpu())
                if ld_kwargs.save_traj:
                    batch_all_frac_coords.append(
                        samples['all_frac_coords'][::down_sample_traj_step].detach().cpu()
                    )
                    batch_all_atom_types.append(
                        samples['all_atom_types'][::down_sample_traj_step].detach().cpu()
                    )

            # collect sampled crystals for this z.
            frac_coords.append(torch.stack(batch_frac_coords, dim=0))
            num_atoms.append(torch.stack(batch_num_atoms, dim=0))
            atom_types.append(torch.stack(batch_atom_types, dim=0))
            lengths.append(torch.stack(batch_lengths, dim=0))
            angles.append(torch.stack(batch_angles, dim=0))
            if ld_kwargs.save_traj:
                all_frac_coords_stack.append(torch.stack(batch_all_frac_coords, dim=0))
                all_atom_types_stack.append(torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (
        frac_coords,
        num_atoms,
        atom_types,
        lengths,
        angles,
        all_frac_coords_stack,
        all_atom_types_stack,
        material_id_list,
    )

def optimization(
    model,
    ld_kwargs,
    data_loader,
    num_starting_points=100,
    num_gradient_steps=5000,
    lr=1e-3,
    num_saved_crys=10,
):
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
        z.requires_grad = True
    else:
        z = torch.randn(
            num_starting_points, model.hparams.hidden_dim, device=model.device
        )
        z.requires_grad = True

    opt = Adam([z], lr=lr)
    model.freeze()

    all_crystals = []
    interval = num_gradient_steps // (num_saved_crys - 1)
    for i in tqdm(range(num_gradient_steps), ncols=79):
        opt.zero_grad()
        loss = model.fc_property(z).mean()
        loss.backward()
        opt.step()

        if i % interval == 0 or i == (num_gradient_steps - 1):
            crystals = model.langevin_dynamics(z, ld_kwargs)
            all_crystals.append(crystals)
    return {
        k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0)
        for k in ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']
    }


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path,
        load_data=('recon' in args.tasks)
        or ('opt' in args.tasks and args.start_from == 'data'),
    )
    print("lattice scaler: ", model.lattice_scaler)
    print(cfg.data.prop)
    print("prop scaler: ", model.prop_scalers)
    prop_scalers = model.prop_scalers

    rel_pressure = 0.0
    rel_spgno = 0.0
    for prop_key, scaler in zip(cfg.data.prop, prop_scalers):
        if prop_key == "pressure":
            # relative pressure
            rel_pressure = (args.pressure - scaler.means.item()) / scaler.stds.item()
        elif prop_key == "spgno":
            rel_spgno = (args.spgno - scaler.means.item()) / scaler.stds.item()
    if args.element_values is not None:
        element_values = list(map(float, args.element_values.split(",")))

    ld_kwargs = SimpleNamespace(
        n_step_each=args.n_step_each,
        step_lr=args.step_lr,
        min_sigma=args.min_sigma,
        save_traj=args.save_traj,
        disable_bar=args.disable_bar,
    )

    if torch.cuda.is_available():
        model.to('cuda')

    if 'recon' in args.tasks:
        print('Evaluate model on the reconstruction task.')
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
        ) = reconstruction(
            test_loader,
            model,
            ld_kwargs,
            args.num_evals,
            args.down_sample_traj_step,
        )

        if args.label == '':
            recon_out_name = 'eval_recon.pt'
        else:
            recon_out_name = f'eval_recon_{args.label}.pt'

        torch.save(
            {
                'eval_setting': args,
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
            model_path / recon_out_name,
        )

    if 'gen' in args.tasks:
        print('Evaluate model on the generation task.')
        start_time = time.time()

        (
            frac_coords,
            num_atoms,
            atom_types,
            lengths,
            angles,
            all_frac_coords_stack,
            all_atom_types_stack,
        ) = generation(
            model,
            ld_kwargs,
            args.num_batches_to_samples,
            args.num_evals,
            args.batch_size,
            args.down_sample_traj_step,
            args.formula,
            args.hydride,
            args.train_data,
            **{
                'energy_per_atom': args.energy_per_atom,
                'energy': args.energy,
                'enthalpy_per_atom': args.enthalpy_per_atom,
                'enthalpy': args.enthalpy,
                'pressure': rel_pressure,
                'spgno': rel_spgno,
                'element_values': element_values,
            },
        )

        if args.label == '':
            gen_out_name = 'eval_gen.pt'
        else:
            gen_out_name = f'eval_gen_{args.label}.pt'
        i = 1
        while Path(model_path / gen_out_name).exists():
            gen_out_name = Path(gen_out_name).stem + f".pt{i}"
            i += 1

        torch.save(
            {
                'eval_setting': args,
                'frac_coords': frac_coords,
                'num_atoms': num_atoms,
                'atom_types': atom_types,
                'lengths': lengths,
                'angles': angles,
                'all_frac_coords_stack': all_frac_coords_stack,
                'all_atom_types_stack': all_atom_types_stack,
                'time': time.time() - start_time,
            },
            model_path / gen_out_name,
        )

    if "target" in args.tasks:
        print('Evaluate model on the target generation task.')
        start_time = time.time()

        (
            frac_coords,
            num_atoms,
            atom_types,
            lengths,
            angles,
            all_frac_coords_stack,
            all_atom_types_stack,
            material_id_list,
        ) = target_generation(
            model=model,
            ld_kwargs=ld_kwargs,
            num_samples_per_z=args.num_evals,
            num_batches_to_samples=args.num_batches_to_samples,
            target_file=args.target_file,
            batch_size=args.batch_size,
            down_sample_traj_step=args.down_sample_traj_step,
        )

        if args.label == '':
            gen_out_name = f'eval_target.pt'
        else:
            gen_out_name = f'eval_target_{args.label}.pt'
        i = 1
        while Path(model_path / gen_out_name).exists():
            gen_out_name = Path(gen_out_name).stem + f".pt{i}"
            i += 1

        torch.save(
            {
                'eval_setting': args,
                'frac_coords': frac_coords,
                'num_atoms': num_atoms,
                'atom_types': atom_types,
                'lengths': lengths,
                'angles': angles,
                'all_frac_coords_stack': all_frac_coords_stack,
                'all_atom_types_stack': all_atom_types_stack,
                'material_id_list': material_id_list,
                'time': time.time() - start_time,
            },
            model_path / gen_out_name,
        )

    if 'opt' in args.tasks:
        print("Unable to do 'opt', skip")
        # print('Evaluate model on the property optimization task.')
        # start_time = time.time()
        # if args.start_from == 'data':
        #     loader = test_loader
        # else:
        #     loader = None
        # optimized_crystals = optimization(model, ld_kwargs, loader)
        # optimized_crystals.update(
        #     {'eval_setting': args, 'time': time.time() - start_time}
        # )

        # if args.label == '':
        #     gen_out_name = 'eval_opt.pt'
        # else:
        #     gen_out_name = f'eval_opt_{args.label}.pt'
        # torch.save(optimized_crystals, model_path / gen_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'target', 'opt'])
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--label', default='')
    parser.add_argument('--formula', help="formula to generate, range is acceptable")
    parser.add_argument('--hydride', action="store_true", help="generate hydride, formula be like (H3-6M0-3M0-3M0-3)1-4")
    parser.add_argument('--train_data', help="sample from trn_cached_data(pkl)")
    parser.add_argument('--target_file', default="target.csv", help="target csv file for task 'target'")
    parser.add_argument(
        '--placeholder',
        help="The above are relative target to std value."
        " Multipy std and add mean results to real target value",
    )
    parser.add_argument(
        '--energy_per_atom', default=-1, type=float, help="relative std, default -1"
    )
    parser.add_argument(
        '--energy', default=-1, type=float, help="relative std, default -1"
    )
    parser.add_argument(
        '--enthalpy_per_atom', default=-1, type=float, help="relative std, default -1"
    )
    parser.add_argument(
        '--enthalpy', default=-1, type=float, help="relative std, default -1"
    )
    parser.add_argument(
        '--pressure', default=0.0, type=float, help="absolute value (GPa), default 0.0"
    )
    parser.add_argument(
        '--spgno', default=1, type=int, help="absolute value of spg number, default 1"
    )  # TODO: change to number range, and record the generated target
    parser.add_argument(
        '--element_values', default=None, help="elem-elem coordinate number for alloy",
    )

    args = parser.parse_args()

    main(args)

