"""Search integer PDM vectors using an energy surrogate (lower energy is better).

Repeated PDM vectors are aggregated by the arithmetic mean of their energies.
A single unique vector defines a constant surrogate. Each returned particle is
its best evaluated position, including the final movement of the swarm.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apollox_common.pdm import METADATA_COLUMNS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g', type=int, required=True)
    parser.add_argument('--target_ratio', type=float, default=0.6)
    parser.add_argument('--min_bound_scale', type=float, default=0.8)
    parser.add_argument('--max_bound_scale', type=float, default=1.2)
    parser.add_argument('--max_iter', type=int, default=100)
    return parser.parse_args()


class EnergyInterpolator:
    def __init__(self, pdm_data, energy_data):
        points = np.asarray(pdm_data, dtype=float)
        energies = np.asarray(energy_data, dtype=float)
        if points.ndim != 2 or not len(points) or not points.shape[1]:
            raise ValueError('PDM data must be a nonempty 2D matrix')
        if energies.shape != (len(points),):
            raise ValueError('One energy is required per PDM vector')
        if not np.isfinite(points).all() or not np.isfinite(energies).all():
            raise ValueError('PDM and energy values must be finite')
        self.points, inverse = np.unique(points, axis=0, return_inverse=True)
        self.energies = np.bincount(inverse, weights=energies) / np.bincount(inverse)
        self.interpolator = (RBFInterpolator(self.points, self.energies, kernel='linear')
                             if len(self.points) > 1 else None)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.points.shape[1] or not np.isfinite(x).all():
            raise ValueError('Prediction PDM vectors have invalid shape or values')
        if self.interpolator is None:
            return np.full(len(x), self.energies[0])
        return self.interpolator(x)


def get_composition_from_poscar(g):
    for file in sorted(Path(f'poscars/generation{g}').glob('POSCAR*')):
        if file.is_file():
            lines = file.read_text().splitlines()
            if len(lines) >= 7:
                elements, counts = lines[5].split(), lines[6].split()
                if len(elements) == len(counts):
                    return ''.join(f'{el}{count}' for el, count in zip(elements, counts))
    raise FileNotFoundError(f'No valid POSCAR found in poscars/generation{g}')


class PSO:
    def __init__(self, dim, x_min, x_max, interpolator, pop_size=30,
                 max_iter=100, w=0.8, c1=2.0, c2=2.0):
        if dim < 1 or pop_size < 1 or max_iter < 0:
            raise ValueError('dim/pop_size must be positive and max_iter nonnegative')
        self.dim, self.pop_size, self.max_iter = dim, pop_size, max_iter
        self.x_min, self.x_max = np.ceil(np.asarray(x_min)), np.floor(np.asarray(x_max))
        if self.x_min.shape != (dim,) or self.x_max.shape != (dim,):
            raise ValueError('Bounds must contain one value per PDM feature')
        if (not np.isfinite(self.x_min).all() or not np.isfinite(self.x_max).all()
                or np.any(self.x_min > self.x_max)):
            raise ValueError('Bounds must contain at least one finite integer per feature')
        self.interpolator, self.w, self.c1, self.c2 = interpolator, w, c1, c2
        self.X = np.round(np.random.uniform(self.x_min, self.x_max, (pop_size, dim)))
        width = self.x_max - self.x_min
        self.V = np.random.uniform(-width, width, (pop_size, dim))
        self.pbest = self.X.copy()
        self.pbest_fitness = self.interpolator.predict(self.X)
        self._update_best()

    def objective_function(self, x):
        return self.interpolator.predict(np.asarray(x).reshape(1, -1))[0]

    def _update_best(self):
        fitness = self.interpolator.predict(self.X)
        better = fitness < self.pbest_fitness
        self.pbest[better] = self.X[better]
        self.pbest_fitness[better] = fitness[better]
        best = np.argmin(self.pbest_fitness)
        self.gbest = self.pbest[best].copy()
        self.gbest_fitness = self.pbest_fitness[best]

    def optimize(self):
        for _ in range(self.max_iter):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            self.V = (self.w * self.V + self.c1 * r1 * (self.pbest - self.X)
                      + self.c2 * r2 * (self.gbest - self.X))
            self.X = np.clip(np.round(self.X + self.V), self.x_min, self.x_max)
            self._update_best()
        return self.pbest.astype(int).copy()


def main(args):
    if not 0 < args.target_ratio <= 1:
        raise ValueError('target_ratio must be in (0, 1]')
    if not 0 <= args.min_bound_scale <= args.max_bound_scale:
        raise ValueError('Bounds scales must satisfy 0 <= min <= max')
    data = pd.read_csv(f'pdm_and_energy/updated_all_structures_summary_batch_{args.g}.csv')
    feature_names = [name for name in data.columns if name not in METADATA_COLUMNS]
    points = data[feature_names].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    energies = pd.to_numeric(data['Energy'], errors='raise').to_numpy(dtype=float)
    surrogate = EnergyInterpolator(points, energies)
    if (points < 0).any():
        raise ValueError('PSO expects unscaled, nonnegative PDM counts')
    optimizer = PSO(points.shape[1], points.min(axis=0) * args.min_bound_scale,
                    points.max(axis=0) * args.max_bound_scale, surrogate,
                    pop_size=max(1, int(len(data) * args.target_ratio)), max_iter=args.max_iter)
    result = pd.DataFrame(optimizer.optimize(), columns=feature_names)
    composition = get_composition_from_poscar(args.g)
    result.insert(0, 'material_id', [f'POSCAR-{composition}-{args.g}_{i}' for i in range(len(result))])
    result.insert(1, 'cif_file', result.material_id + '.cif')
    Path('temp').mkdir(exist_ok=True)
    result.to_csv(f'temp/optimized_particles_distribution_with_ids_{args.g}.csv', index=False)


if __name__ == '__main__':
    main(parse_args())
