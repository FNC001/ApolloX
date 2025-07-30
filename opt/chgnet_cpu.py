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
    """
    Run structure optimization for a single file.

    Args:
        name (Path): Path to the input structure file.
        pstress (float): Pressure stress to apply.
        model_name (str): Name of the model being used.
        mlp_trajname (Path): Path to save the optimization trajectory.
        fmax (float): Force convergence criterion.
        mlp_optstep (int): Maximum number of optimization steps.
        calc (Calculator): ASE calculator object.
        output_dir (Path): Directory to save output files.

    Returns:
        dict or None: A dictionary with the structure name and final energy,
                      or None if optimization failed or was skipped.
    """
    verbose = True
    stream = sys.stdout if verbose else io.StringIO()
    with contextlib.redirect_stdout(stream):
        print(f"Start to Optimize Structure using {model_name}")
        start = time.time()

        atoms = read(name)

        # Check for atoms that are too close to each other
        dis_mtx = atoms.get_all_distances(mic=True)
        row, col = np.diag_indices_from(dis_mtx)
        dis_mtx[row, col] = np.nan
        min_dis = np.nanmin(dis_mtx)

        if min_dis > 0.6:
            atoms.calc = calc
            # Convert pressure from GPa to eV/Å^3
            aim_stress = pstress * 0.006242
            ucf = UnitCellFilter(atoms, scalar_pressure=aim_stress)

            opt = FIRE(ucf, trajectory=str(mlp_trajname))
            opt.run(fmax=fmax, steps=mlp_optstep)

            energy = float(atoms.get_potential_energy())

            # Save optimized structure
            output_file = output_dir / f"{name.name}.optdone"
            write(output_file, atoms, format='vasp')

            print(f"Optimization done! Cost {time.time() - start:.2f} seconds.")
            return {"name": name.name, "energy": energy}
        else:
            warnings.warn(f"The minimum distance of two atoms in {name.name} is {min_dis}, which is too close. Skipping.")
            return None


def main():
    """
    Main function to parse arguments and run optimizations.
    """
    parser = argparse.ArgumentParser(description="Optimize structures using CHGNet.")
    parser.add_argument("--input_pattern", type=str, default="POSCAR*",
                        help="Glob pattern to match input structure files.")
    parser.add_argument("--mlp_optstep", type=int, default=200,
                        help="Maximum number of optimization steps.")
    parser.add_argument("--fmax", type=float, default=0.01,
                        help="Force convergence criterion in eV/Å.")
    parser.add_argument("--output_csv", type=str, default="sorted_energies.csv",
                        help="Path to save the sorted energies CSV.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save optimized structures and trajectories.")
    args = parser.parse_args()

    # Find input files that haven't been optimized yet
    input_files = sorted([f for f in Path().glob(args.input_pattern) if not f.name.endswith(".optdone")])

    if not input_files:
        print(f"No files matched the input pattern: '{args.input_pattern}'")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CHGNet model...")
    chgnet = CHGNet.load()
    # Set the calculator to use the CPU
    calc = MLP(chgnet, use_device="cpu")
    print("Model loaded. Starting optimizations on CPU.")

    results = []
    for name in input_files:
        print(f"---- Optimizing {name} ----")
        # Generate a unique trajectory file name for each structure
        traj_name = output_dir / f"{name.stem}.traj"
        result = run_opt(
            name=name,
            pstress=0,
            model_name="CHGNet",
            mlp_trajname=traj_name,
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
        output_csv_path = args.output_csv
        df_sorted.to_csv(output_csv_path, columns=['name', 'energy'], index=False)
        print(f"\nEnergy summary saved to {output_csv_path}")
    else:
        print("⚠️ No valid results were generated to save.")


if __name__ == "__main__":
    main()
