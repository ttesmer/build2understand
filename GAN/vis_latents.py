import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

GEN_NOISE_DIM=50

# Generate a random normal tensor
noise = torch.randn(1, GEN_NOISE_DIM)

# Convert to NumPy array for visualization
data = noise.squeeze(0).numpy()

# Plot the distribution
sns.histplot(data, bins=30, kde=True)

# Show the plot
plt.show()
