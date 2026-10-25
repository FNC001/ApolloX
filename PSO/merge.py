"""Join fresh structure descriptors to exactly one finite energy per structure."""
import numpy as np
import pandas as pd


def main():
    structures = pd.read_csv('all_structures_summary.csv')
    energies = pd.read_csv('sorted_energies.csv')
    if structures.empty or energies.empty:
        raise ValueError('Structure and energy tables must be nonempty')
    energies['material_id'] = energies['name'].astype(str) + '.optdone'
    if structures.material_id.duplicated().any() or energies.material_id.duplicated().any():
        raise ValueError('Structure IDs and energy IDs must be unique')
    merged = structures.merge(energies[['material_id', 'energy']], on='material_id', how='left', validate='one_to_one')
    merged = merged.rename(columns={'energy': 'Energy'})
    if not np.isfinite(pd.to_numeric(merged.Energy, errors='raise')).all():
        raise ValueError('Every structure must have a finite matching energy')
    merged.to_csv('Merged_all_structures_with_energy.csv', index=False)


if __name__ == '__main__':
    main()
