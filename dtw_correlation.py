import os

import pandas as pd
from tqdm import tqdm

files = [f"filtered_Turbine_Data_Penmanshiel_{i:02d}_2022-01-01_-_2023-01-01.csv"
         for i in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

power_list = []
for item in tqdm(files):
    path = os.path.join('.materials/Penmanshiel_SCADA_2022_WT01-15/', item)
    if os.path.exists(path):
        data = pd.read_csv(path)
        # Only select the 'Power (kW)' column
        data_power = data[['Power (kW)']]
        # Append to list
        power_list.append(data_power)

# Calculate Dynamic Time Warping (DTW) distance between each pair of turbines
from dtaidistance import dtw
num_turbines = len(power_list)
dtw_matrix = pd.DataFrame(index=[f"Turbine_{i+1:02d}" for i in range(num_turbines)],
                          columns=[f"Turbine_{i+1:02d}" for i in range(num_turbines)])
for i in range(num_turbines):
    for j in range(num_turbines):
        if i == j:
            dtw_matrix.iloc[i, j] = 0.0
        elif pd.isna(dtw_matrix.iloc[i, j]):
            distance = dtw.distance(power_list[i]['Power (kW)'].values,
                                    power_list[j]['Power (kW)'].values)
            dtw_matrix.iloc[i, j] = distance
            dtw_matrix.iloc[j, i] = distance  # Symmetric matrix
# Draw the heatmap of DTW distance matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 10))
sns.heatmap(dtw_matrix.astype(float), annot=True, fmt=".2f", cmap='viridis', cbar=True)
plt.title('Dynamic Time Warping (DTW) Distance Matrix')
plt.tight_layout()
plt.show()
