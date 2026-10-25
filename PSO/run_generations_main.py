"""Run the full V2 PSO workflow, stopping immediately when any stage fails."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PSO.runtime import repository_root, load_config, run_python, validate_training_artifacts


def main():
    root, config = repository_root(), load_config()
    validate_training_artifacts(config)
    generations = config['gen_num']
    if not isinstance(generations, int) or generations < 1:
        raise ValueError('gen_num must be a positive integer')
    run_python(root / 'PSO/initial_structures.py')
    for generation in range(1, generations + 1):
        run_python(root / 'PSO/submit_one_gen.py', '--g', generation)
        run_python(root / 'PSO/submit_opt.py', '--g', generation)
    print(f'{generations} generations completed successfully. Results: pdm_and_energy/ and poscars/.')


if __name__ == '__main__':
    main()
