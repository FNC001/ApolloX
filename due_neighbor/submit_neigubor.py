"""Run bounded foreground neighbor workers and propagate failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys


def submit_jobs(start, end, max_cores=64, schema_path=None):
    if max_cores < 1 or start > end:
        raise ValueError('Require positive concurrency and start <= end')
    script = Path(__file__).with_name('run_neighbor.py').resolve()
    folders = [Path(f'split_{index}').resolve() for index in range(start, end + 1)]
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(folder)
    schema_path = str(Path(schema_path).expanduser().resolve()) if schema_path else None

    def run(folder):
        command = [sys.executable, str(script), '--input_dir', str(folder)]
        if schema_path:
            command += ['--schema', schema_path]
        with (folder / 'run.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    with ThreadPoolExecutor(max_workers=max_cores) as executor:
        list(executor.map(run, folders))
    print('All specified tasks completed successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=64)
    parser.add_argument('--max-workers', type=int, default=64)
    parser.add_argument('--schema', help='Shared training schema for all split folders')
    args = parser.parse_args()
    submit_jobs(args.start, args.end, args.max_workers, args.schema)
