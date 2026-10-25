"""Run dataset preparation sequentially, stopping on the first failed stage."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def generate_dataset(config_path):
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    root = Path(__file__).resolve().parents[1]
    generation_type = config.get('generation_type', '').strip().lower()
    if generation_type not in {'single', 'variable'}:
        raise ValueError("generation_type must be single or variable")
    dataset = Path(config['dataset_path']).expanduser().resolve()
    initial = Path(config['initial_structure']).expanduser()
    if not initial.is_absolute():
        initial = root / 'original_structures' / initial
    modes = shlex.split(config['mode']) if isinstance(config['mode'], str) else config['mode']
    python = sys.executable
    scripts = root / 'prepare_dataset'
    commands = [
        [python, str(root / f'generate_structure/bulk/generate_{generation_type}_component.py'),
         '--input', str(initial), '--num', str(config['num']), '--outdir', str(dataset)],
        [python, str(scripts / 'compute_pdm.py'), '--input_dir', str(dataset / 'poscar'),
         '--cutoff', str(config['cutoff']), '--n_jobs', str(config['n_jobs']), '--mode', *modes,
         '--starts_with', 'POSCAR', '--output_csv', str(dataset / 'all_structures_summary.csv')],
        [python, str(scripts / 'transfer_POSCAR_to_cif.py'), str(dataset), '--n', str(config['n_jobs'])],
        [python, str(scripts / 'split_dataset.py'), '--input', str(dataset / 'all_structures_summary.csv'),
         '--output_dir', str(dataset), '--train_ratio', str(config['train_ratio']),
         '--test_ratio', str(config['test_ratio']), '--val_ratio', str(config['val_ratio'])],
        [python, str(scripts / 'append_cif.py'), '--input_csvs',
         *[str(dataset / f'{name}_set_scaled.csv') for name in ['train', 'test', 'val']],
         '--output_names', 'train', 'test', 'val', '--cif_dir', str(dataset / 'cif'), '--output_dir', str(dataset)],
        [python, str(scripts / 'convert_CVS.py'), '--input_dir', str(dataset)],
    ]
    for command in commands:
        print('[Running] ' + shlex.join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    generate_dataset(parser.parse_args().config)
