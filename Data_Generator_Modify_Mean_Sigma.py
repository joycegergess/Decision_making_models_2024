import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_trials = 10000
decision_boundary = 0  # Decision boundary (b)

# Randomly generate 1000 different trials which all differ in how far the mean is to the decision boundary
means = np.random.uniform(-3, 3, num_trials)

# Randomly generate 1000 different trials which all differ in how big the standard deviation is
sigma = np.random.uniform(0.01, 4, num_trials)

# Generate a percept for each trial based on its mean and standard deviation
percepts = np.random.normal(means, sigma)

# Determine if this percept was correct or incorrect
# A percept is correct if the sample is on the same side of the decision boundary as its mean
percepts_correct = (percepts > decision_boundary) == (means > decision_boundary)

# Create a 2D matrix using the means and sigma as the dimensions
# Bin the means and sigma
mean_bins = np.linspace(-3, 3, 40)
sigma_bins = np.linspace(0.01, 4, 40)

# Convert the continuous data (means & sigma) into their discrete bins
mean_digitized = np.digitize(means, mean_bins)
sigma_digitized = np.digitize(sigma, sigma_bins)

# Create empty 2D matrix to store p(correct)
# np.zeros(rows, columns)
matrix = np.zeros((len(mean_bins), len(sigma_bins)))

# Calculate p(correct) for each bin
# Note: loop starts at 1 because np.digitize returns indices starting from 1 (not 0)
for i in range(1, len(mean_bins)):
    for j in range(1, len(sigma_bins)):
        bin_indices = (mean_digitized == i) & (sigma_digitized == j)
        # bin_indices boolean array will only be 'True' when both conditions are met
        if np.sum(bin_indices) > 0:
        # Verify if there are any trials in current bin of loop
            matrix[i-1, j-1] = np.mean(percepts_correct[bin_indices])
            # Reminder: i and j start at 1 but the 'means' and 'sigma' arrays begin at 0
            # Calculate mean of elements in 'percepts_correct' where 'bin_indices' is 'True'

# Plot the 2D matrix
plt.figure(figsize=(10, 8))
plt.imshow(matrix, extent=[sigma_bins[0], sigma_bins[-1], mean_bins[0], mean_bins[-1]], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='P(Correct)')
plt.xlabel('Noise (SD)')
plt.ylabel('Discriminability (Mean)')
plt.title('P(Correct) as a Function of Noise and Discriminability')
plt.show()