import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the JSON files
data_directory = "piece_states_removal"
# Lists to store the data
data = []

# Iterate through each file in the directory
for file_name in os.listdir(data_directory):
    file_path = os.path.join(data_directory, file_name)
    with open(file_path, "r") as file:
        # Load JSON data from file and extend the data list
        data.extend(json.load(file))

# Convert the data list to a DataFrame
df = pd.DataFrame(data)

# Filter out rows where states_removed is not higher than 0
df = df[df['states_removed'] > 0]

# Group by piece_type and calculate mean and std
mean_df = df.groupby('piece_type').agg({'states_removed': ['mean'], 'percentage_states_removed': ['mean']})
std_df = df.groupby('piece_type').agg({'states_removed': ['std'], 'percentage_states_removed': ['std']})

# Combine mean and std
mean_df.columns = ['states_removed_mean', 'percentage_states_removed_mean']
std_df.columns = ['states_removed_std', 'percentage_states_removed_std']
combined_df = pd.concat([mean_df, std_df], axis=1)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot mean and std for states_removed
combined_df[['states_removed_mean', 'states_removed_std']].plot.bar(ax=ax[0])
ax[0].set_title("Mean and Standard Deviation of States Removed by Piece Type")
ax[0].set_xlabel("Piece Type")
ax[0].set_ylabel("States Removed")

# Plot mean and std for percentage_states_removed
combined_df[['percentage_states_removed_mean', 'percentage_states_removed_std']].plot.bar(ax=ax[1])
ax[1].set_title("Mean and Standard Deviation of Percentage States Removed by Piece Type")
ax[1].set_xlabel("Piece Type")
ax[1].set_ylabel("Percentage States Removed")

plt.tight_layout()
plt.show()