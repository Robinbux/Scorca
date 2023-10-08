import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

ENTROPY_DIR = '/Users/robinbux/Desktop/RBC_New/experiments/entropy_comp'
CUTOFF = 25

# Initialize dictionaries to store data
naive_entropy_differences = defaultdict(list)
optimized_entropy_differences = defaultdict(list)

# Loop through all json files in the directory
for filename in os.listdir(ENTROPY_DIR):
    with open(f'{ENTROPY_DIR}/{filename}', 'r') as f:
        data = json.load(f)
        state_counts = np.array(data['state_counts'])
        if not len(state_counts):
            continue
        differences = state_counts[:, 0] - state_counts[:, 1]

        # Add differences to the corresponding list
        if data['sense_strategy'] == 'naive_entropy':
            for i, diff in enumerate(differences):
                naive_entropy_differences[i].append(diff)
        elif data['sense_strategy'] == 'optimized_entropy':
            for i, diff in enumerate(differences):
                optimized_entropy_differences[i].append(diff)

# Calculate means and stds
naive_entropy_means = [np.mean(diffs) for diffs in naive_entropy_differences.values()][:CUTOFF]
naive_entropy_stds = [np.std(diffs) if np.std(diffs) > 0 else 0.1 for diffs in naive_entropy_differences.values()][:CUTOFF]
optimized_entropy_means = [np.mean(diffs) for diffs in optimized_entropy_differences.values()][:CUTOFF]
optimized_entropy_stds = [np.std(diffs) if np.std(diffs) > 0 else 0.1 for diffs in optimized_entropy_differences.values()][:CUTOFF]




# Create x-axis values for each plot
x_naive = list(range(1, len(naive_entropy_means) + 1))
x_optimized = list(range(1, len(optimized_entropy_means) + 1))

# Define colors for naive and optimized entropy
naive_color = 'blue'
optimized_color = 'red'

# Plot naive entropy means and fill between mean +/- std
plt.plot(x_naive, naive_entropy_means, color=naive_color, label='Naive entropy')
plt.fill_between(x_naive,
                 np.subtract(naive_entropy_means, naive_entropy_stds),
                 np.add(naive_entropy_means, naive_entropy_stds),
                 color=naive_color, alpha=0.2)

# Plot optimized entropy means and fill between mean +/- std
plt.plot(x_optimized, optimized_entropy_means, color=optimized_color, label='Adapted entropy')
plt.fill_between(x_optimized,
                 np.subtract(optimized_entropy_means, optimized_entropy_stds),
                 np.add(optimized_entropy_means, optimized_entropy_stds),
                 color=optimized_color, alpha=0.2)
# Add title, labels and legend
plt.title('Mean and Std of removed states after scan \nfor Naive and Adapted Entropy strategy')
plt.xlabel('Turns')
plt.ylabel('Removed states after scan')
plt.legend()

# Set the y scale to log
plt.yscale('log')

# Show the plot
plt.show()
