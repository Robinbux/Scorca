import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = [
    [-0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000],
    [-0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000],
    [-0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000],
    [-0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000],
    [ 0.27620,  0.27620,  0.27620,  0.27620,  0.27620,  0.27620,  0.27620,  0.27620],
    [ 0.54895,  0.27620,  0.54895,  0.27620,  0.27620,  0.54895,  0.27620,  0.54895],
    [ 0.45372,  0.45372,  0.45372,  0.45372,  0.45372,  0.45372,  0.45372,  0.45372],
    [-0.00000,  0.45372, -0.00000, -0.00000, -0.00000, -0.00000,  0.45372, -0.00000],
]

# Flipping the data horizontally
data = np.flip(data, axis=0)
data_rounded = np.round(data, 2)


# Creating labels for the plot
x_labels = list('ABCDEFGH')
y_labels = list(map(str, range(1, 9)))
# Creating the heatmap
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(data_rounded, annot=True, fmt=".2f", linewidths=.5, ax=ax, cmap="Reds", cbar=False, xticklabels=x_labels, yticklabels=y_labels)

# Setting the X axis labels on the top
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# Hiding axis labels
ax.set_xlabel('')
ax.set_ylabel('')

# Displaying the plot
plt.show()