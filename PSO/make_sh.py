"""Generate explicit PSO evaluation jobs without truncating PDM vectors."""
import argparse
import math
from pathlib import Path
import shlex
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PSO.runtime import repository_root
from apollox_common.pdm import METADATA_COLUMNS


def find_apollox_path():
    return str(repository_root())


def get_composition_from_poscar(g):
    from PSO.pso_optimizer import get_composition_from_poscar as composition
    return composition(g)


def generate_scripts(g, apollox_path=None):
    root = Path(apollox_path) if apollox_path is not None else repository_root()
    formula = get_composition_from_poscar(g)
    data = pd.read_csv(f'temp/standardized_optimized_particles_distribution{g}.csv')
    if data.empty or 'material_id' not in data or data.material_id.isna().any() or data.material_id.duplicated().any():
        raise ValueError('Generation CSV requires nonempty rows with unique material_id values')
    columns = [column for column in data if column not in METADATA_COLUMNS]
    if not columns:
        raise ValueError('No PDM feature columns found in the standardized CSV')
    values = data[columns].apply(pd.to_numeric, errors='raise')
    if not all(math.isfinite(value) for value in values.to_numpy().flat):
        raise ValueError('PDM features must contain only finite numeric values')
    output = Path('temp') / f'sh_files_{g}'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to reuse old generation scripts: {output}')
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, row in data.iterrows():
        label = str(row.material_id)
        if not label.strip() or Path(label).name != label or label in {'.', '..'}:
            raise ValueError('material_id must be a nonempty filename without directory components')
        vector = ','.join(map(str, values.loc[index]))
        command = [sys.executable, str(root / 'cond-cdvae/scripts/evaluate.py'), '--model_path', str(Path.cwd()),
                   '--tasks', 'gen', f'--formula={formula}', '--pressure=0', f'--label={label}',
                   f'--element_values={vector}', '--batch_size=1', '--num_batches_to_samples=1']
        script = output / f'run_evaluation_{index}.sh'
        script.write_text('#!/bin/bash\nset -euo pipefail\n' + shlex.join(command) + '\n')
        jobs.append({'script': script, 'label': label, 'output': Path(f'eval_gen_{label}.pt')})
    return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g', type=int, required=True)
    generate_scripts(parser.parse_args().g)
