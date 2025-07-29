import pandas as pd
import numpy as np
import argparse

# Set up command-line argument parsing.
def parse_args():
    parser = argparse.ArgumentParser(description="Standardize the optimized particle distribution data.")
    parser.add_argument('--g', type=int, required=True, help="The passed parameter g, which is used to read the data file.")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments.
    args = parse_args()
    g = args.g  # Retrieve the passed g parameter.

    # Mean and standard deviation. Replace them with the mean and standard deviation from the scaler_stats.txt file generated during model training.
    means = np.array([14.433375, 31.44525, 31.495625, 31.428375, 31.467125, 157.296875,
                      14.4625, 31.52275, 31.416, 31.4295, 157.2615, 14.389,
                      31.43425, 31.4445, 157.324875, 14.443875, 31.446875, 157.38675,
                      14.41325, 157.3855, 386.67225])

    std_devs = np.array([3.04619617, 4.50080575, 4.50263599, 4.51435154, 4.55583903, 7.87113335,
                         3.08825902, 4.4979976, 4.60447543, 4.53624622, 7.92454212, 3.04974245,
                         4.47240729, 4.51682076, 7.81332716, 3.13657456, 4.53061008, 7.88550407,
                         3.12333547, 7.9248432, 8.74691831])

    # Read the csv file.
    df = pd.read_csv(f'./optimized_particles_distribution_with_ids_{g}.csv')

    # Extract the data to be standardized (from the second column to the second-to-last column).
    data_to_standardize = df.iloc[:, 1:-1]

    # Standardize data： (x - mean) / std
    standardized_data = (data_to_standardize - means) / std_devs

    # Replace the original data with the standardized data.
    df.iloc[:, 1:-1] = standardized_data

    # Save as a new CSV file.
    output_file_path = f'./standardized_optimized_particles_distribution{g}.csv'
    df.to_csv(output_file_path, index=False)

    print(f"The standardized data has been saved to {output_file_path}")
