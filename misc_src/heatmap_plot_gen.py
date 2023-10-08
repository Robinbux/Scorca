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

# Creating the heatmap
fig, ax = plt.subplots(figsize=(12, 10))

# Using a lambda function to adjust the font weight to bold and size for the annotations
cax = sns.heatmap(data_rounded, annot=True, fmt=".2f", linewidths=.5, ax=ax, cmap="Reds", cbar=False, xticklabels=False, yticklabels=False)
for text in cax.texts:
    text.set_weight('bold')
    text.set_size(15)
    if text.get_text() == "-0.00":
        text.set_text("0")

# Setting the X axis labels on the top
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# Hiding axis labels
ax.set_xlabel('')
ax.set_ylabel('')

# Displaying the plot
plt.show()
