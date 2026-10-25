"""Generate one PSO population, requiring each job's exact fresh output."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PSO.runtime import repository_root, load_config, run_python, require_new_paths, validate_training_artifacts
from PSO.make_sh import generate_scripts


def validate_single_export(extracted):
    gen_dir = Path(extracted) / 'gen'
    manifest_path = gen_dir / 'structure_manifest.csv'
    manifest = pd.read_csv(manifest_path, dtype={'filename': str, 'material_id': str})
    fields = {'filename', 'material_id', 'eval_index', 'sample_index'}
    if not fields <= set(manifest) or len(manifest) != 1 or manifest[list(fields)].isna().any().any():
        raise ValueError(f'Expected a complete one-sample export manifest in {gen_dir}')
    row = manifest.iloc[0]
    files = sorted(path.name for path in gen_dir.glob('*.vasp'))
    if files != [row.filename] or Path(row.filename).name != row.filename or row.eval_index != 0 or row.sample_index != 0:
        raise ValueError(f'Export files and single-sample manifest do not match in {gen_dir}')
    if not row.material_id.strip() or (gen_dir / row.filename).stat().st_size == 0:
        raise ValueError(f'Empty sample identity or structure in {gen_dir}')


def generate_one_generation(g):
    root, config = repository_root(), load_config()
    validate_training_artifacts(config)
    output = Path('temp') / f'pt_files_{g}'
    raw_csv = Path('temp') / f'optimized_particles_distribution_with_ids_{g}.csv'
    scaled_csv = Path('temp') / f'standardized_optimized_particles_distribution{g}.csv'
    require_new_paths([output, raw_csv, scaled_csv, Path('temp') / f'sh_files_{g}'])
    params = config['PSO']
    run_python(root / 'PSO/pso_optimizer.py', '--g', g, '--target_ratio', params['target_ratio'],
               '--min_bound_scale', params['min_bound_scale'], '--max_bound_scale', params['max_bound_scale'],
               '--max_iter', params['max_iter'])
    run_python(root / 'prepare_dataset/standardize.py', '--input', raw_csv,
               '--scaler', Path(config['scaler_path']).expanduser().resolve(), '--output', scaled_csv)
    jobs = generate_scripts(g, root)
    for job in jobs:
        require_new_paths([job['output'], job['output'].with_suffix('')])
    output.mkdir(parents=True)
    for job in jobs:
        subprocess.run(['bash', str(job['script'])], check=True)
        expected = job['output']
        if not expected.is_file():
            raise RuntimeError(f'Generation succeeded without producing its expected output: {expected}')
        run_python(root / 'cond-cdvae/scripts/extract_gen.py', expected)
        extracted = expected.with_suffix('')
        validate_single_export(extracted)
        shutil.move(str(expected), output / expected.name)
        shutil.move(str(extracted), output / extracted.name)
    print(f'Generation {g}: completed {len(jobs)} generation jobs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g', type=int, required=True)
    generate_one_generation(parser.parse_args().g)


if __name__ == '__main__':
    main()
