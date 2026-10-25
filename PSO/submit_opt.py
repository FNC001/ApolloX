"""Relax one generation; a failed or missing job aborts the generation."""
import argparse
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


def run_task(gen_dir, structure_name, chgnet_script, chgnet_cfg, pdm_script, pdm_cfg, apollox_root, schema_args=None):
    gen_dir = Path(gen_dir).resolve()
    expected = gen_dir / (structure_name + '.optdone')
    output_csv = pdm_cfg.get('output_csv', 'all_structures_summary.csv')
    if output_csv != 'all_structures_summary.csv':
        raise ValueError('PSO pdm.output_csv must be all_structures_summary.csv')
    require_new_paths([expected, gen_dir / 'sorted_energies.csv', gen_dir / output_csv,
                       gen_dir / 'Merged_all_structures_with_energy.csv'])
    run_python(chgnet_script, '--input_pattern', structure_name,
               '--mlp_optstep', chgnet_cfg.get('mlp_optstep', 1),
               '--fmax', chgnet_cfg.get('fmax', 0.01), '--output_csv', 'sorted_energies.csv', cwd=gen_dir)
    if not expected.is_file() or not (gen_dir / 'sorted_energies.csv').is_file():
        raise RuntimeError(f'Optimization did not produce fresh structure and energy outputs in {gen_dir}')
    if schema_args is None:
        raise ValueError('The training PDM schema is required for PSO relaxation')
    run_python(pdm_script, '--input_dir', '.', *schema_args, '--output_csv', output_csv, cwd=gen_dir)
    run_python(Path(apollox_root) / 'PSO/merge.py', cwd=gen_dir)
    merged = pd.read_csv(gen_dir / 'Merged_all_structures_with_energy.csv')
    if merged.empty or not np.isfinite(pd.to_numeric(merged['Energy'], errors='raise')).all():
        raise ValueError(f'Missing or nonfinite relaxed energies in {gen_dir}')
    return gen_dir / 'Merged_all_structures_with_energy.csv'


def run_chgnet_parallel(base_dir, chgnet_script, chgnet_cfg, pdm_script, pdm_cfg,
                        apollox_root, max_workers=2, min_free_mem_gb=4.0, schema_args=None):
    tasks = []
    for folder in sorted(Path(base_dir).glob('eval_gen_*')):
        if not folder.is_dir():
            continue
        gen_dir = folder / 'gen'
        files = sorted(gen_dir.glob('*.vasp'))
        if len(files) != 1:
            raise ValueError(f'Expected exactly one generated structure in {gen_dir}; got {len(files)}')
        structure_name = folder.name[len('eval_gen_'):] + '.vasp'
        new_path = gen_dir / structure_name
        if files[0] != new_path:
            if new_path.exists():
                raise FileExistsError(new_path)
            files[0].rename(new_path)
        tasks.append((gen_dir, structure_name))
    if not tasks:
        raise ValueError(f'No generation tasks found in {base_dir}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for gen_dir, structure_name in tasks:
            while psutil.virtual_memory().available / 1e9 < min_free_mem_gb:
                print('[MEMORY] Waiting for memory before submitting another optimization', flush=True)
                time.sleep(10)
            futures.append(executor.submit(run_task, gen_dir, structure_name, chgnet_script,
                                           chgnet_cfg, pdm_script, pdm_cfg, apollox_root, schema_args))
        # Calling result propagates failures; no backup-only success path exists.
        return [future.result() for future in futures]


def copy_selected_poscars(apollox_root, base_dir, g, combined_csv_path):
    frame = pd.read_csv(combined_csv_path)
    destination = Path.cwd() / 'poscars' / f'generation{g + 1}'
    require_new_paths([destination])
    sources = []
    for material_id in frame.material_id.astype(str):
        if Path(material_id).name != material_id:
            raise ValueError('Structure IDs must be filenames without directory components')
        candidates = [Path.cwd() / 'pso_initial/poscar' / material_id]
        candidates += [folder / 'gen' / material_id for folder in Path(base_dir).glob('eval_gen_*') if folder.is_dir()]
        found = [path for path in candidates if path.is_file()]
        if len(found) != 1:
            raise FileNotFoundError(f'Expected exactly one optimized file for {material_id}; found {len(found)}')
        sources.append(found[0])
    destination.mkdir(parents=True)
    for source in sources:
        shutil.copy2(source, destination / source.name)


def run_chgnet_and_pdm(g):
    base_dir = Path('temp') / f'pt_files_{g}'
    if not base_dir.is_dir():
        raise FileNotFoundError(base_dir)
    root = repository_root()
    config = load_config()
    schema = validate_training_artifacts(config)
    output = Path('pdm_and_energy') / f'updated_all_structures_summary_batch_{g + 1}.csv'
    require_new_paths([output, Path('poscars') / f'generation{g + 1}'])
    opt = config['opt']
    script_name = config['opt_script']
    if Path(script_name).name != script_name or not (root / 'opt' / script_name).is_file():
        raise ValueError('opt_script must name an existing script in opt/')
    paths = run_chgnet_parallel(base_dir, root / 'opt' / script_name, opt,
                                root / 'prepare_dataset/compute_pdm.py', config['pdm'], root,
                                max_workers=opt.get('max_workers', 2), min_free_mem_gb=opt.get('min_free_mem_gb', 4.0),
                                schema_args=pdm_arguments(config))
    combined = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if combined.material_id.duplicated().any():
        raise ValueError('Duplicate generated structure IDs')
    n = config['structure_num_per_gen']
    if n < 1 or len(combined) > n:
        raise ValueError('Generated population exceeds the configured positive generation size')
    reserve_path = Path('pso_initial/reserve.csv')
    reserve = pd.read_csv(reserve_path)
    missing = n - len(combined)
    if len(reserve) < missing:
        raise ValueError(f'Not enough unused initial structures: need {missing}, have {len(reserve)}')
    final = pd.concat([combined, reserve.head(missing)], ignore_index=True)
    from apollox_common.pdm import align_features
    align_features(final, schema, allow_missing=False)
    if final.material_id.duplicated().any() or not np.isfinite(pd.to_numeric(final.Energy)).all():
        raise ValueError('Next generation contains duplicate IDs or invalid energies')
    output.parent.mkdir(exist_ok=True)
    staged = output.with_suffix('.pending.csv')
    require_new_paths([staged])
    final.to_csv(staged, index=False)
    copy_selected_poscars(root, base_dir, g, staged)
    staged.replace(output)
    reserve.iloc[missing:].to_csv(reserve_path, index=False)
    print(f'Generation {g + 1}: saved {len(final)} valid structures to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g', type=int, required=True)
    run_chgnet_and_pdm(parser.parse_args().g)
