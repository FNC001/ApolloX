#!/usr/bin/env python3
import io
import sys
import time
import json
import argparse
import warnings
import contextlib
from pathlib import Path

import pandas as pd
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE

try:
    from ase.filters import UnitCellFilter
except ImportError:
    from ase.constraints import UnitCellFilter

from chgnet.model import CHGNet
from chgnet.model import CHGNetCalculator as MLP


def run_opt(name, pstress, model_name, mlp_trajname, fmax, mlp_optstep, calc, output_dir):
    verbose = True
    stream = sys.stdout if verbose else io.StringIO()
    with contextlib.redirect_stdout(stream):
        print("Start to Optimize Structures using", model_name)
        start = time.time()

        atoms = read(name)

        dis_mtx = atoms.get_all_distances(mic=True)
        row, col = np.diag_indices_from(dis_mtx)
        dis_mtx[row, col] = np.nan
        min_dis = np.nanmin(dis_mtx)

        if min_dis > 0.6:
            atoms.calc = calc
            aim_stress = pstress * 0.006242
            ucf = UnitCellFilter(atoms, scalar_pressure=aim_stress)

            opt = FIRE(ucf, trajectory=mlp_trajname)
            opt.run(fmax=fmax, steps=mlp_optstep)

            energy = float(atoms.get_potential_energy())

            # Save optimized structure
            output_file = output_dir / f"{name.name}.optdone"
            write(output_file, atoms, format='vasp')

            print(f"Optimization done! Cost {time.time() - start:.2f} seconds.")
            return {"name": name.name, "energy": energy}
        else:
            warnings.warn(f"The minimum distance of two atoms is {min_dis}, too close.")
            return None


def main():
    parser = argparse.ArgumentParser(description="Optimize structures using CHGNet.")
    parser.add_argument("--input_pattern", type=str, default="POSCAR*",
                        help="Glob pattern to match input structure files.")
    parser.add_argument("--mlp_optstep", type=int, default=20,
                        help="Maximum number of optimization steps.")
    parser.add_argument("--fmax", type=float, default=0.01,
                        help="Force convergence criterion in eV/Å.")
    parser.add_argument("--output_csv", type=str, default="sorted_energies.csv",
                        help="Path to save the sorted energies CSV.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save optimized structures and trajectories.")
    args = parser.parse_args()

    input_files = sorted([f for f in Path().glob(args.input_pattern) if not f.name.endswith(".optdone")])

    if not input_files:
        print(f"No files matched pattern: {args.input_pattern}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chgnet = CHGNet.load()
    calc = MLP(chgnet, use_device="cuda:0")

    results = []
    for name in input_files:
        print(f"---- Optimizing {name} ----")
        result = run_opt(
            name=name,
            pstress=0,
            model_name="CHGNET",
            mlp_trajname="traj.traj",
            fmax=args.fmax,
            mlp_optstep=args.mlp_optstep,
            calc=calc,
            output_dir=output_dir,
        )
        if result:
            results.append(result)

    if results:
        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by='energy')
        df_sorted.to_csv(args.output_csv, columns=['name', 'energy'], index=False)
        print(f"Energy summary saved to {args.output_csv}")
    else:
        print("⚠️  No valid results to save.")


if __name__ == "__main__":
    main()
