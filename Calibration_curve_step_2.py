import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_trials = 10000
sigma = 1.0  # Standard deviation (perceptual noise remains same for all trials)
decision_boundary = 0  # Decision boundary (b)

# Randomly generate 1000 different means (trials) which all differ in how far they are to the decision boundary
means = np.random.uniform(-3, 3, num_trials)

# Generate a sample for each trials based on its mean
samples = np.random.normal(means, sigma)

# Determine whether the percept is correct
# A percept is correct if the sample is on the same side of the decision boundary as its mean
percepts_correct = (samples > decision_boundary) == (means > decision_boundary)

# Calculate absolute distance of each sample from decision-boundary, this will be my x-axis
distance_from_boundary = np.abs(samples - decision_boundary)

# Define the bin edges for the 20-point bins
bin_edges = np.linspace(0, 4, 21)

# Arrays to store bin centers and confidence values
bin_centers = []
confidence_values = []

# Calculate confidence for each bin
for i in range(len(bin_edges) - 1):
    # Find indices of percepts that fall into the current bin
    bin_indices = np.where((distance_from_boundary >= bin_edges[i]) & (distance_from_boundary < bin_edges[i + 1]))[0]
        # np.where returns percepts from each bin depending on their distance from DB
        # bin_edges[i + 1] represents upper boundary of current bin
        # purpose of [0] here is to get the actual array of indices from the tuple returned by np.where

    # Get samples and outcomes for the current bin using indices
    bin_samples = samples[bin_indices]
    bin_percepts_correct = percepts_correct[bin_indices]

    # Calculate center of the bins to know where to plot the points on x-axis
    bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        # this caclulates midpoint of current bin
    bin_centers.append(bin_center)
        # this appends (adds) the calculated midpoints (bin_center) to the bin_centers array created above

    # Calculate the confidence: proportion of correct outcomes
    if len(bin_percepts_correct) > 0:
        # this code allows me to check if bin_outcomes is not empty (has at least one generated percept)
        # if bin has outcomes, can calculate the mean
        confidence = np.mean(bin_percepts_correct)  # Outcomes are binary (0 and 1), mean is proportion of 1s (correct outcomes)
    else:
        confidence = 0  # If the bin has no outcomes, set confidence to 0

    confidence_values.append(confidence)
    # adds calculated confidence values to confidence_values list

# Convert lists to arrays
bin_centers = np.array(bin_centers)
confidence_values = np.array(confidence_values)

# Iterate through bin centers and confidence values
for center, confidence in zip(bin_centers, confidence_values):
    print(f"Average Bin Mean: {center:.2f}, Confidence Value: {confidence:.2f}")

# Plotting
plt.figure(figsize=(10, 10))
plt.plot(bin_centers, confidence_values, 'o-')
plt.axhline(y=0.5, color='gray', linestyle='--')

plt.xlabel('Distance to DB')
plt.ylabel('Confidence')
plt.ylim(0, 1)
plt.title('Confidence vs. Distance to Decision Boundary')
plt.show()
