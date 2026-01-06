import os

import pandas as pd
from tqdm import tqdm

files = [f"filtered_Turbine_Data_Penmanshiel_{i:02d}_2022-01-01_-_2023-01-01.csv"
         for i in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

corr_matrixs = []
for item in tqdm(files):
    path = os.path.join('.materials/Penmanshiel_SCADA_2022_WT01-15/', item)
    if os.path.exists(path):
        data = pd.read_csv(path)
        # Remove the "date" column
        if 'date' in data.columns:
            data = data.drop(columns=['date'])
        elif 'Time' in data.columns:
            data = data.drop(columns=['Time'])
        elif 'time' in data.columns:
            data = data.drop(columns=['time'])
        # Calculate the Pearson Correlation Coefficient for it
        corr_matrix = data.corr(method='pearson')
        corr_matrixs.append(corr_matrix)

# Average the correlation matrices
avg_corr_matrix = sum(corr_matrixs) / len(corr_matrixs)
# Draw the heatmap
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 10))
sns.heatmap(avg_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Average Pearson Correlation Coefficient Matrix')
plt.tight_layout()
plt.show()

# Check correlations with target variable "Power (kW)"
# Correlation of features with 'Power (kW)':
# Power (kW)                                                 1.000000
# Wind speed (m/s)                                           0.885401
# Density adjusted wind speed (m/s)                          0.885024
# Wind speed Sensor 2 (m/s)                                  0.881841
# Wind speed Sensor 1, Minimum (m/s)                         0.880037
# Wind speed Sensor 1 (m/s)                                  0.877565
# Wind speed, Minimum (m/s)                                  0.875632
# Wind speed Sensor 2, Minimum (m/s)                         0.866098
# Wind speed, Maximum (m/s)                                  0.861178
# Wind speed Sensor 2, Maximum (m/s)                         0.858901
# Wind speed Sensor 1, Maximum (m/s)                         0.851190
# Wind speed, Standard deviation (m/s)                       0.709003
# Wind speed Sensor 2, Standard deviation (m/s)              0.693680
# Stator temperature 1, Min (°C)                             0.690348
# Stator temperature 1 (°C)                                  0.689398
# Stator temperature 1, Max (°C)                             0.685869
# Wind speed Sensor 1, Standard deviation (m/s)              0.646400
# Gear oil temperature, Standard deviation (°C)              0.503696
# Time-based IEC B.2.2 (Users View)                          0.502721
# Gear oil temperature, Max (°C)                             0.411432
# Gear oil temperature (°C)                                  0.379704
# Gear oil temperature, Min (°C)                             0.349770
# Ambient temperature (converter), StdDev (°C)               0.278535
# Time-based System Avail.                                   0.169859
# Time-based Contractual Avail. (Global)                     0.168415
# Time-based IEC B.2.3 (Users View)                          0.168232
# Stator temperature 1, StdDev (°C)                          0.161277
# Time-based IEC B.2.4 (Users View)                          0.156470
# Wind direction, Minimum (°)                                0.151766
# Time-based Contractual Avail. (Custom)                     0.148056
# Time-based IEC B.3.2 (Manufacturers View)                  0.134719
# Wind direction (°)                                         0.123443
# Nacelle position, Minimum (°)                              0.121579
# Nacelle position (°)                                       0.119231
# Nacelle position, Maximum (°)                              0.115359
# Time-based Contractual Avail.                              0.110951
# Wind direction, Maximum (°)                                0.081677
# Nacelle position, Standard deviation (°)                  -0.034849
# Lost Production (Time-based IEC B.3.2) (kWh)              -0.100227
# Nacelle ambient temperature, StdDev (°C)                  -0.103023
# Ambient temperature (converter), Max (°C)                 -0.111035
# Lost Production (Time-based IEC B.2.4) (kWh)              -0.117288
# Ambient temperature (converter) (°C)                      -0.117357
# Wind direction, Standard deviation (°)                    -0.117838
# Lost Production to Downtime and Curtailment Total (kWh)   -0.118639
# Lost Production to Downtime (kWh)                         -0.118665
# Lost Production (Time-based IEC B.2.2) (kWh)              -0.119714
# Lost Production (Time-based IEC B.2.3) (kWh)              -0.119714
# Nacelle temperature, Standard deviation (°C)              -0.121637
# Ambient temperature (converter), Min (°C)                 -0.123785
# Nacelle ambient temperature, Min (°C)                     -0.209571
# Nacelle ambient temperature (°C)                          -0.210827
# Nacelle ambient temperature, Max (°C)                     -0.211535
# Nacelle temperature, Min (°C)                             -0.508738
# Nacelle temperature (°C)                                  -0.521521
# Nacelle temperature, Max (°C)                             -0.529292
# Time-based System Avail. (Planned)                              NaN
# Name: Power (kW), dtype: float64
target_corr = avg_corr_matrix['Power (kW)'].sort_values(ascending=False)
print("Correlation of features with 'Power (kW)':")
print(target_corr)
