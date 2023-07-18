import numpy as np
import matplotlib.pyplot as plt

from src.utils import convert_centipawn_score_to_win_probability

# Generate scores from -1000 to 1000
scores = np.linspace(-1000, 1000, 100)

# Specify different values of k to analyze
k_values = [1, 2, 4, 8, 16]

# Plot win probability for each value of k
plt.figure(figsize=(10, 6))

for k in k_values:
    win_probabilities = [convert_centipawn_score_to_win_probability(score, k) for score in scores]
    plt.plot(scores, win_probabilities, label=f'k={k}')

# Add labels and title
plt.xlabel('Score (centipawns)')
plt.ylabel('Win Probability')
plt.title('Win Probability vs. Score for Different Values of k')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()