import numpy as np
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Optimizing particle distribution with PSO")
    parser.add_argument('--g', type=int, required=True, help="The input parameter g, which is used to read the data file.")
    return parser.parse_args()

def objective_function(x):
    return 0

class PSO:
    def __init__(self,
                 dim,
                 x_min,
                 x_max,
                 pop_size=30,
                 max_iter=100,
                 w=0.8,
                 c1=2.0,
                 c2=2.0):

        self.dim = dim
        self.x_min = x_min
        self.x_max = x_max
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.X = np.random.uniform(self.x_min, self.x_max, (self.pop_size, self.dim))
        self.V = np.random.uniform(-abs(self.x_max - self.x_min), abs(self.x_max - self.x_min),
                                   (self.pop_size, self.dim))
        self.pbest = self.X.copy()
        self.pbest_fitness = np.array([objective_function(x) for x in self.X])
        gbest_index = np.argmin(self.pbest_fitness)
        self.gbest = self.X[gbest_index].copy()
        self.gbest_fitness = self.pbest_fitness[gbest_index]

    def optimize(self):
        for t in range(self.max_iter):
            fitness = np.array([objective_function(x) for x in self.X])
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
    data = pd.read_csv(f'./updated_all_structures_summary_batch_{g}.csv')

    numeric_data = data.select_dtypes(include=[np.number])
    min_bounds = numeric_data.min() * 0.8
    max_bounds = numeric_data.max() * 1.2
    target_size = int(len(data) * 0.6)
    dim = numeric_data.shape[1]
    x_min = min_bounds.values
    x_max = max_bounds.values

    # Create a PSO optimizer instance
    pso = PSO(dim=dim, x_min=x_min, x_max=x_max, pop_size=target_size, max_iter=100)

    # Run optimization
    final_positions = pso.optimize()

    # Convert the optimization results into a DataFrame.
    result_df = pd.DataFrame(final_positions, columns=numeric_data.columns)

    # Add the material_id column
    result_df['material_id'] = [f'POSCAR-B12Mo12Co12Fe12Ni12O60-{g}_' + str(i) for i in range(result_df.shape[0])]

    # Rearrange the column order to place material_id at the front.
    result_df = result_df[['material_id'] + list(numeric_data.columns)]

    # Save as a csv file.
    output_file_path = f'./optimized_particles_distribution_with_ids_{g}.csv'
    result_df.to_csv(output_file_path, index=False)

    print(f"All optimized particle distributions have been saved to {output_file_path}")
