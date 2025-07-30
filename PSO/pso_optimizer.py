import numpy as np
import pandas as pd
import argparse
import os
import re
from scipy.interpolate import RBFInterpolator
def parse_args():
    parser = argparse.ArgumentParser(description="Optimizing particle distribution with PSO")
    parser.add_argument('--g', type=int, required=True, help="The input parameter g, which is used to read the data file.")
    parser.add_argument('--target_ratio', type=float, default=0.6, help="Ratio for target population size (default: 0.6)")
    parser.add_argument('--min_bound_scale', type=float, default=0.8, help="Scaling factor for minimum bounds (default: 0.8)")
    parser.add_argument('--max_bound_scale', type=float, default=1.2, help="Scaling factor for maximum bounds (default: 1.2)")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations (default: 100)")
    return parser.parse_args()

class EnergyInterpolator:
    def __init__(self, pdm_data, energy_data):
        """
        pdm_data: Input PDM data (2D array)
        energy_data: Input energy data (1D array)
        """
        self.interpolator = RBFInterpolator(pdm_data, energy_data, kernel='linear')

    def predict(self, x):
        """
        x: Particle positions (2D array)
        Returns: Estimated energy values
        """
        return self.interpolator(x)
def get_composition_from_poscar(g):
    poscar_dir = f'./poscars/generation{g}/'
    for file in os.listdir(poscar_dir):
        if file.startswith('POSCAR'):
            with open(os.path.join(poscar_dir, file), 'r') as f:
                lines = f.readlines()
                # Assuming line 6 contains the element symbols
                if len(lines) > 5:
                    elements = lines[5].strip().split()
                    # Assuming line 7 contains the number of atoms per element
                    counts = lines[6].strip().split()
                    if len(elements) == len(counts):
                        composition = ''.join(f'{elem}{count}' for elem, count in zip(elements, counts))
                        return composition
    return None  # Fallback to default if no POSCAR found

class PSO:
    def __init__(self,
                 dim,
                 x_min,
                 x_max,
                 interpolator,
                 pop_size=30,
                 max_iter=100,
                 w=0.8,
                 c1=2.0,
                 c2=2.0):

        self.dim = dim
        self.x_min = x_min
        self.x_max = x_max
        self.interpolator=interpolator
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.X = np.random.uniform(self.x_min, self.x_max, (self.pop_size, self.dim))
        self.V = np.random.uniform(-abs(self.x_max - self.x_min), abs(self.x_max - self.x_min),
                                   (self.pop_size, self.dim))
        self.pbest = self.X.copy()
        self.pbest_fitness = np.array([self.objective_function(x) for x in self.X])
        gbest_index = np.argmin(self.pbest_fitness)
        self.gbest = self.X[gbest_index].copy()
        self.gbest_fitness = self.pbest_fitness[gbest_index]
    def objective_function(self,x):
        return self.interpolator.predict(x.reshape(1,-1))[0]
    def optimize(self):
        for t in range(self.max_iter):
            fitness = np.array([self.objective_function(x) for x in self.X])
            better_mask = fitness < self.pbest_fitness
            self.pbest[better_mask] = self.X[better_mask]
            self.pbest_fitness[better_mask] = fitness[better_mask]
            current_gbest_index = np.argmin(self.pbest_fitness)
            if self.pbest_fitness[current_gbest_index] < self.gbest_fitness:
                self.gbest = self.pbest[current_gbest_index].copy()
                self.gbest_fitness = self.pbest_fitness[current_gbest_index]

            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            self.V = self.w * self.V \
                     + self.c1 * r1 * (self.pbest - self.X) \
                     + self.c2 * r2 * (self.gbest - self.X)

            self.X = self.X + self.V
            self.X = np.clip(self.X, self.x_min, self.x_max)
            self.X = np.round(self.X).astype(int)

        return self.X


if __name__ == "__main__":
    args = parse_args()
    g = args.g
    target_ratio = args.target_ratio
    min_bound_scale = args.min_bound_scale
    max_bound_scale = args.max_bound_scale
    max_iter = args.max_iter

    data = pd.read_csv(f'./pdm_and_energy/updated_all_structures_summary_batch_{g}.csv')

    # Extract composition from POSCAR file
    composition = get_composition_from_poscar(g)

    numeric_data = data.select_dtypes(include=[np.number])
    energy_data=numeric_data['Energy']
    pdm_data=numeric_data.drop(columns=['Energy'])
    min_bounds = pdm_data.min() * min_bound_scale
    max_bounds = pdm_data.max() * max_bound_scale
    target_size = int(len(data) * target_ratio)
    dim = pdm_data.shape[1]
    x_min = min_bounds.values
    x_max = max_bounds.values
    interpolator = EnergyInterpolator(pdm_data.values, energy_data.values)
    # Create a PSO optimizer instance
    pso = PSO(dim=dim, x_min=x_min, x_max=x_max, interpolator=interpolator,pop_size=target_size, max_iter=max_iter)

    # Run optimization
    final_positions = pso.optimize()

    # Convert the optimization results into a DataFrame
    result_df = pd.DataFrame(final_positions, columns=pdm_data.columns)

    # Add the material_id column
    result_df['material_id'] = [f'POSCAR-{composition}-{g}_' + str(i) for i in range(result_df.shape[0])]
    result_df['cif_file'] = result_df['material_id'] + '.cif'

    # Rearrange the column order to place material_id at the front
    result_df = result_df[['material_id', 'cif_file'] + list(pdm_data.columns)]

    # Ensure output directory exists
    os.makedirs('./temp', exist_ok=True)

    # Save as a csv file
    output_file_path = f'./temp/optimized_particles_distribution_with_ids_{g}.csv'
    result_df.to_csv(output_file_path, index=False)

    print(f"All optimized particle distributions have been saved to {output_file_path}")
