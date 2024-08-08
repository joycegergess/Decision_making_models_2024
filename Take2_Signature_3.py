# Take 2 for signature 3 (psychometric curve).
# This time we will not measure p(correct).
# Instead, we will measure p(choosing right) regardless of whether it is the correct choice or not.
# Goal of this plot is just to examine how choice behavior changes based on evidence.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_trials = 10000
sigma = 1.0  # Standard deviation (perceptual noise)
decision_boundary = 0  # Decision boundary (b)

# Randomly sample 10000 means (trials) which all differ in how far they are to the decision boundary
means = np.linspace(-4, 4, num_trials)

# Generate a random percept for each of these means
percepts = np.random.normal(means, sigma)

# Calculate distance of each mean from decision-boundary, this will be my x-axis, how discriminable is the given stimulus (x)
distance_from_boundary = means - decision_boundary

# Define bins for the full x-axis range
bins = np.linspace(-4, 4, 25)

# Define confidence masks (you had these variables at the end in your earlier code)
low_conf_mask = (np.abs(means - decision_boundary) <= 0.5)
high_conf_mask = (np.abs(means - decision_boundary) > 0.5)

# Initialize lists to store results
bin_centers = []
prob_choosing_right_low_conf = []
prob_choosing_right_high_conf = []

for i in range(len(bins) - 1):
    # Define the bin range
    bin_mask = (distance_from_boundary >= bins[i]) & (distance_from_boundary < bins[i + 1])

    # Calculate bin center for the point on the figure
    bin_center = (bins[i] + bins[i + 1]) / 2
    bin_centers.append(bin_center)

    # Low confidence trials
    low_conf_bin_mask = bin_mask & low_conf_mask

    # High confidence trials
    high_conf_bin_mask = bin_mask & high_conf_mask

