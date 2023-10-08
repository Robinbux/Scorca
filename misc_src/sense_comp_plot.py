import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# Directories to compare
dirs = ['../sense_comp/all_possible_states', '../sense_comp/likely_senses', '../sense_comp/opp_move_weight']
# Corresponding labels for the legend
labels = ['All possible states', 'Likely senses', 'Opp move weights']

# Initialize a figure
plt.figure(figsize=(10, 6))

# Iterate over directories
for dir, label in zip(dirs, labels):
    # Initialize empty dict to store all second values
    second_values_dict = {}

    # Iterate over each json file
    for filename in glob.glob(f'{dir}/*'):
        with open(filename, 'r') as f:
            data = json.load(f)

        # Extract second values from 'state_counts' and add to dict
        for i, state_count in enumerate(data['state_counts']):
            if i not in second_values_dict:
                second_values_dict[i] = [state_count[1]]
            else:
                second_values_dict[i].append(state_count[1])

    # Calculate mean and standard deviation for each index
    mean_values = np.array([np.mean(v) for v in second_values_dict.values()])
    std_dev_values = np.array([np.std(v) for v in second_values_dict.values()])

    # Generate index for second values
    indices = np.array(list(second_values_dict.keys()))

    # Plot mean values with a line and fill between lines for standard deviation
    plt.plot(indices, mean_values, label=label)
    plt.fill_between(indices, mean_values - std_dev_values, mean_values + std_dev_values, alpha=0.2)

# Labels, title and legend
plt.xlabel('Turn')
plt.ylabel('States after sense')
#plt.title('Mean and Standard Deviation of Second Values in state_counts Across Directories')
plt.legend()

plt.ylim(bottom=1)  # Set the lower limit for y-axis to 1
plt.yscale('log')  # Set the y-axis to a log scale
plt.xticks(np.arange(min(indices), max(indices)+1, 2))  # Set x-axis ticks to use whole integers in steps of 2
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
