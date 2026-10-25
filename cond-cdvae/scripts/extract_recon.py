# Reconstruction matching with explicit eval/sample identity

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
from extract_gen import extract_pt, atoms_from_batch, evaluation_records


def write_match(i, name, recon_i, matchers, atoms_dir):
    ser = pd.Series({"fname": f"{i}.*.vasp", "mp_id": name,
                     "eval_index": recon_i["eval_index"], "sample_index": recon_i["sample_index"]})
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
    ser = pd.Series({"fname": f"{i}.*.vasp", "mp_id": name,
                     "eval_index": recon_i["eval_index"], "sample_index": recon_i["sample_index"]})
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

    # Standard export is shared with gen/target, including all evaluations.
    extract_pt(recon_pt_file)
    recon = torch.load(recon_pt_file, map_location='cpu', weights_only=False)
    records = evaluation_records(recon)
    true_list = atoms_from_batch(recon["input_data_batch"])
    if len(true_list) != recon['num_atoms'].shape[1]:
        raise ValueError("Ground truth must have one structure for each sample axis entry")
    recon_records = [
        {"material_id": record["material_id"], "eval_index": record["eval_index"],
         "sample_index": record["sample_index"], "recon_atoms": record["atoms"],
         "true_atoms": true_list[record["sample_index"]]}
        for record in records
    ]

    matchers = get_matchers()
    if original_matcher:
        serlist = Parallel(njobs, backend="multiprocessing")(
            delayed(write_match_ori)(i, recon_i["material_id"], recon_i, matchers, atoms_dir)
            for i, recon_i in tqdm(enumerate(recon_records), ncols=79, total=len(recon_records))
        )
    else:
        serlist = Parallel(njobs, backend="multiprocessing")(
            delayed(write_match)(i, recon_i["material_id"], recon_i, matchers, atoms_dir)
            for i, recon_i in tqdm(enumerate(recon_records), ncols=79, total=len(recon_records))
        )
    df = pd.DataFrame(serlist)
    with open(extract_dir / (recon_pt_file.name + ".matches.pkl"), "wb") as f:
        pickle.dump(recon_records, f)
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
