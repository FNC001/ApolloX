# match a prediction of extracted pkl from `evaluate.py --tasks target --target_file [...]`
# with target file accroding to "material_id" and corresponding feather "cif".
import argparse
import pickle
import warnings
from pathlib import Path

import pandas as pd
import torch
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from statgen import get_matchers, to_format_table
from tqdm import tqdm
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predict_pt", help="predict pt file (before extract) for target material_id")
    parser.add_argument("-P", "--predict_pkl", help="predict pkl file (after extract) for Atoms list")
    parser.add_argument("-t", "--target_feather", help="the predict target feather file with material_id and cif")
    args = parser.parse_args()
    return args


def series2structure(series):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        structure = Structure.from_str(series.cif, fmt='cif')
    return structure


def match_one(material_id, predict_atoms, target_df, matchers):
    predict = AseAtomsAdaptor.get_structure(predict_atoms)
    target = series2structure(target_df.query(f"material_id == '{material_id}'").iloc[0])
    ser = pd.Series(name=material_id)
    for mat_name, matcher in matchers.items():
        fit = matcher.fit(target, predict)
        dist = matcher.get_rms_dist(target, predict) if fit else (pd.NA, pd.NA)
        ser[mat_name] = fit
        ser[f"{mat_name}_normrms"] = dist[0]
        ser[f"{mat_name}_maxrms"] = dist[1]
    return ser


def main(predict_pt, predict_pkl, target_feather):
    predict_dict = torch.load(predict_pt)
    print(len(predict_dict["material_id_list"]))
    if len(predict_dict["material_id_list"]) == 0:
        raise ValueError("No material_id_list in predict pt, refuse to continue.")
    material_id_list = predict_dict["material_id_list"]
    with open(predict_pkl, "rb") as f:
        predict_atoms_list = pickle.load(f)
    target_df = pd.read_feather(target_feather)
    # assert len(predict_atoms_list) == len(material_id_list), "number of predicts must equal targets"
    missing_ids = set(material_id_list) - set(target_df.material_id.to_list())
    assert len(missing_ids) == 0, f"{missing_ids} are missing in target_feather"

    matchers = get_matchers()

    serlist = []
    for material_id, predict_atoms in tqdm(zip(material_id_list, predict_atoms_list), total=len(predict_atoms_list)):
        predict = AseAtomsAdaptor.get_structure(predict_atoms)
        target = series2structure(target_df.query(f"material_id == '{material_id}'").iloc[0])
        ser = pd.Series(name=material_id)
        for mat_name, matcher in matchers.items():
            fit = matcher.fit(target, predict)
            dist = matcher.get_rms_dist(target, predict) if fit else (pd.NA, pd.NA)
            ser[mat_name] = fit
            ser[f"{mat_name}_normrms"] = dist[0]
            ser[f"{mat_name}_maxrms"] = dist[1]
        serlist.append(ser)
    data = pd.DataFrame(serlist)
    data.index.name = "material_id"

    print(data)
    with open(Path(predict_pkl).with_name(f"{Path(predict_pt).stem}.match.table"), "w") as f:
        f.write(to_format_table(data))



if __name__ == "__main__":
    args = parse_args()
    main(args.predict_pt, args.predict_pkl, args.target_feather)
