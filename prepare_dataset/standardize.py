import pandas as pd
import numpy as np

# Your original data
data = pd.read_csv('optimized_particles_distribution.csv')

#Average values and standard deviations. Replace them with the data in “scaler_stats.txt”.
means = np.array([9.384, 20.607125, 20.581875, 20.49, 20.635125, 102.7655,
                  9.3925, 20.508875, 20.60475, 20.5675, 102.859625, 9.421875,
                  20.485375, 20.567875, 102.943125, 9.434375, 20.597125,
                  102.621375, 9.397125, 102.89125, 252.243625])

stds = np.array([3.00125707, 4.46298938, 4.41444181, 4.44374842, 4.37032507, 10.49995285,
                 2.93192663, 4.46432204, 4.4044327, 4.42514901, 10.50560183, 3.02384631,
                 4.36489245, 4.41490577, 10.58399217, 2.96478386, 4.38312865, 10.37733194,
                 3.00460758, 10.44713709, 15.98889527])


data = data.drop(columns=['energy'], errors='ignore')


standardized_data = (data - means) / stds

standardized_data.to_csv('standardized_optimized_particles_distribution.csv', index=False)

print("done")
