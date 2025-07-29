# extract recon pt
# NOT finished

import pickle
from pathlib import Path

import click
import torch
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write
from joblib import Parallel, delayed
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from statgen import get_matchers, to_format_table
from tqdm import tqdm


def write_match(i, name, recon_i, matchers, atoms_dir):
    ser = pd.Series({"fname": f"{i}.*.vasp", "mp_id": name})
    recon_st = AseAtomsAdaptor.get_structure(recon_i["recon_atoms"])
    true_st = AseAtomsAdaptor.get_structure(recon_i["true_atoms"])
    for matcher_name, matcher in matchers.items():
        fit = matcher.fit(true_st, recon_st)
        ser[matcher_name] = fit
        ser[matcher_name + "avgd"] = matcher.get_rms_dist(true_st, recon_st)[0] if fit else pd.NA
    CifWriter(recon_st, symprec=0.01, refine_struct=False).write_file(atoms_dir.joinpath(f"{i}.recon.cif"))
    CifWriter(true_st, symprec=0.01, refine_struct=False).write_file(atoms_dir.joinpath(f"{i}.true.cif"))
    write(atoms_dir.joinpath(f"{i}.recon.vasp"), recon_i["recon_atoms"])
    write(atoms_dir.joinpath(f"{i}.true.vasp"), recon_i["true_atoms"])
    return ser


def write_match_ori(i, name, recon_i, matchers, atoms_dir):
    ser = pd.Series({"fname": f"{i}.*.vasp", "mp_id": name})
    recon_st = AseAtomsAdaptor.get_structure(recon_i["recon_atoms"])
    true_st = AseAtomsAdaptor.get_structure(recon_i["true_atoms"])
    for matcher_name, matcher in matchers.items():
        rms_dist = matcher.get_rms_dist(recon_st, true_st)
        if rms_dist is None:
            fit = False
            rms_dist = pd.NA
        else:
            fit = True
            rms_dist = rms_dist[0]
        ser[matcher_name] = fit
        ser[matcher_name + "avgd"] = rms_dist
    CifWriter(recon_st, symprec=0.01, refine_struct=False).write_file(atoms_dir.joinpath(f"{i}.recon.cif"))
    CifWriter(true_st, symprec=0.01, refine_struct=False).write_file(atoms_dir.joinpath(f"{i}.true.cif"))
    write(atoms_dir.joinpath(f"{i}.recon.vasp"), recon_i["recon_atoms"])
    write(atoms_dir.joinpath(f"{i}.true.vasp"), recon_i["true_atoms"])
    return ser


def extract_recon_pt(recon_pt_file: Path, njobs, original_matcher=False):
    if original_matcher:
        extract_dir = recon_pt_file.with_name(f"{recon_pt_file.stem}.ori")
    else:
        extract_dir = recon_pt_file.with_name(recon_pt_file.stem)

    pkl = extract_dir.joinpath(recon_pt_file.name + ".pkl")
    idx = extract_dir.joinpath(recon_pt_file.name + ".idx")
    atoms_dir = extract_dir.joinpath(recon_pt_file.name + ".dir")
    atoms_dir.mkdir(exist_ok=True, parents=True)

    recon = torch.load(recon_pt_file)

    from eval_utils import get_crystals_list  # change working dir

    batch_idx = 0
    crys_array_list = get_crystals_list(
        recon["frac_coords"][batch_idx],
        recon["atom_types"][batch_idx],
        recon["lengths"][batch_idx],
        recon["angles"][batch_idx],
        recon["num_atoms"][batch_idx],
    )
    true_crystal_array_list = get_crystals_list(
        recon["input_data_batch"].frac_coords,
        recon["input_data_batch"].atom_types,
        recon["input_data_batch"].lengths,
        recon["input_data_batch"].angles,
        recon["input_data_batch"].num_atoms,
    )
    recon_list = [
        Atoms(
            c['atom_types'],
            scaled_positions=c['frac_coords'],
            cell=c['lengths'].tolist() + c['angles'].tolist(),
        )
        for c in crys_array_list
    ]
    true_list = [
        Atoms(
            c['atom_types'],
            scaled_positions=c['frac_coords'],
            cell=c['lengths'].tolist() + c['angles'].tolist(),
        )
        for c in true_crystal_array_list
    ]

    recon_dict = {
        name: {"recon_atoms": recon_c, "true_atoms": true_c}
        for name, recon_c, true_c in zip(recon["input_data_batch"].mp_id, recon_list, true_list)
    }

    matchers = get_matchers()
    with open(pkl, "wb") as f:
        pickle.dump(recon_dict, f)
    if original_matcher:
        serlist = Parallel(njobs, backend="multiprocessing")(
            delayed(write_match_ori)(i, name, recon_i, matchers, atoms_dir)
            for i, (name, recon_i) in tqdm(enumerate(recon_dict.items()), ncols=79, total=len(recon_dict))
        )
    else:
        serlist = Parallel(njobs, backend="multiprocessing")(
            delayed(write_match)(i, name, recon_i, matchers, atoms_dir)
            for i, (name, recon_i) in tqdm(enumerate(recon_dict.items()), ncols=79, total=len(recon_dict))
        )
    df = pd.DataFrame(serlist)
    with open(idx, "w") as f:
        f.write(to_format_table(df))


@click.command
@click.argument("recon_pt_file")
@click.option("-j", "--njobs", default=16)
@click.option("--ori", "original_matcher", is_flag=True, help="use CDVAE original matcher, default False")
def save_recon_structure(recon_pt_file, njobs, original_matcher):
    recon_pt_file = Path(recon_pt_file).resolve()

    extract_recon_pt(recon_pt_file, njobs, original_matcher=original_matcher)


if __name__ == "__main__":
    save_recon_structure()
