"""Create and relax a fresh initial population without reusing prior results."""
import concurrent.futures
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PSO.runtime import repository_root, load_config, run_python, pdm_arguments, require_new_paths, validate_training_artifacts


def run_chgnet_for_structure(structure_file, chgnet_script, chgnet_cfg, work_root):
    folder = Path(work_root) / (structure_file.name + '_dir')
    require_new_paths([folder])
    folder.mkdir()
    shutil.copy2(structure_file, folder / structure_file.name)
    run_python(chgnet_script, '--input_pattern', structure_file.name,
               '--mlp_optstep', chgnet_cfg.get('mlp_optstep', 1), '--fmax', chgnet_cfg.get('fmax', 0.01),
               '--output_csv', 'sorted_energies.csv', cwd=folder)
    csv_path = folder / 'sorted_energies.csv'
    optimized = folder / (structure_file.name + '.optdone')
    if not csv_path.is_file() or not optimized.is_file():
        raise RuntimeError(f'Optimization did not produce fresh outputs for {structure_file.name}')
    shutil.copy2(optimized, Path(work_root) / optimized.name)
    return csv_path


def run_parallel_chgnet(output_dir, chgnet_script, chgnet_cfg, max_workers=2, min_free_mem_gb=4.0):
    structures = sorted(path for path in Path(output_dir).glob('POSCAR*') if path.is_file() and not path.name.endswith('.optdone'))
    if not structures:
        raise ValueError('No initial structures were generated')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for structure in structures:
            while psutil.virtual_memory().available / 1e9 < min_free_mem_gb:
                print('[MEMORY] Waiting before submitting another optimization', flush=True)
                time.sleep(10)
            futures.append(executor.submit(run_chgnet_for_structure, structure, chgnet_script, chgnet_cfg, output_dir))
        return [future.result() for future in futures]


def main():
    root, config = repository_root(), load_config()
    validate_training_artifacts(config)
    n, generations = config['structure_num_per_gen'], config['gen_num']
    if n < 1 or generations < 1:
        raise ValueError('structure_num_per_gen and gen_num must be positive')
    ratio = config['PSO']['target_ratio']
    if not 0 < ratio <= 1:
        raise ValueError('PSO target_ratio must be in (0, 1]')
    total = n + generations * (n - max(1, int(n * ratio)))
    pso_dir = root / 'PSO'
    initial_dir = Path.cwd() / 'pso_initial'
    output = initial_dir / 'poscar'
    reserve = initial_dir / 'reserve.csv'
    require_new_paths([Path('temp'), Path('pdm_and_energy'), Path('poscars'), initial_dir])
    script_name = config['opt_script']
    if Path(script_name).name != script_name or not (root / 'opt' / script_name).is_file():
        raise ValueError('opt_script must name an existing script in opt/')
    run_python(root / 'generate_structure/bulk/generate_single_component.py', '--input',
               root / 'original_structures' / config['poscar_name'], '--num', total, '--outdir', initial_dir)
    opt = config['opt']
    files = run_parallel_chgnet(output, root / 'opt' / script_name, opt,
                               max_workers=opt.get('max_workers', 2), min_free_mem_gb=opt.get('min_free_mem_gb', 4.0))
    energies = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if len(energies) != total or not np.isfinite(pd.to_numeric(energies.energy)).all():
        raise ValueError('Every initial structure must have a finite energy')
    energies.to_csv(output / 'sorted_energies.csv', index=False)
    run_python(root / 'prepare_dataset/compute_pdm.py', '--input_dir', output,
               *pdm_arguments(config), '--output_csv', output / 'all_structures_summary.csv')
    run_python(pso_dir / 'merge.py', cwd=output)
    population = pd.read_csv(output / 'Merged_all_structures_with_energy.csv')
    if len(population) != total:
        raise ValueError('Incomplete initial population')
    initial = population.iloc[:n]
    Path('pdm_and_energy').mkdir()
    destination = Path('poscars/generation1')
    destination.mkdir(parents=True)
    for material_id in initial.material_id:
        shutil.copy2(output / material_id, destination / material_id)
    initial.to_csv('pdm_and_energy/updated_all_structures_summary_batch_1.csv', index=False)
    population.iloc[n:].to_csv(reserve, index=False)
    print(f'Prepared {n} initial structures and {len(population) - n} reserve structures')


if __name__ == '__main__':
    main()
