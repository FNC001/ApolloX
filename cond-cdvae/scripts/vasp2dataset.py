import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
from ase.io import read, write
from tqdm import tqdm


def atoms2cifstring(atoms):
    with io.BytesIO() as buffer, redirect_stdout(buffer):
        write('-', atoms, format='cif')
        cif = buffer.getvalue().decode()  # byte to string
    return cif


def read_outcar(outcar):
    try:
        atoms = read(outcar, format="vasp-out")
    except Exception:
        print(outcar, "NOT complete")
        return False
    with open(outcar, "r") as f:
        for line in f.readlines():
            if "enthalpy is  TOTEN" in line:
                enthalpy0 = float(line.strip().split()[4])
                break
    with open(outcar, "r") as f:
        for line in f.readlines()[::-1]:
            if "enthalpy is  TOTEN" in line:
                enthalpy = float(line.strip().split()[4])
                break
    _id = Path(outcar).stem  # *.OUTCAR_ -> *
    return {
            "material_id": _id,
            "formula": atoms.get_chemical_formula("metal"),
            "natoms": len(atoms),
            "volume": atoms.get_volume(),
            "volume_per_atom": atoms.get_volume() / len(atoms),
            # "energy": energy,
            # "energy_per_atom": energy / len(atoms),
            "enthalpy": enthalpy,
            "enthalpy_per_atom": enthalpy / len(atoms),
            "enthalpy0": enthalpy0,
            "enthalpy0_per_atom": enthalpy0 / len(atoms),
            "cif": atoms2cifstring(atoms),
        }


def main(dir_list):
    ser_list = []
    for d in dir_list:
        for f in tqdm(Path(d).glob("*.OUTCAR")):
            data_dict = read_outcar(f)
            if data_dict:
                ser = pd.Series(data_dict)
            ser_list.append(ser)
    df = pd.DataFrame(ser_list)
    return df


if __name__ == "__main__":
    # <dir>/*.OUTCAR
    dir_list = sys.argv[1:]
    df = main(dir_list)
    df.to_feather("opt.feather")
