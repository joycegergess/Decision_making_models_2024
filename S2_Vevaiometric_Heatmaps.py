import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_trials = 10000
decision_boundary = 0  # Decision boundary (b)

# Randomly generate 10000 different trials which all differ in how far the mean is to the decision boundary
means = np.random.uniform(-3, 3, num_trials)

# Randomly generate 10000 different trials which all differ in how big the standard deviation is
sigma = np.random.uniform(0.01, 4, num_trials)

# Generate a percept for each trial based on its mean and standard deviation
percepts = np.random.normal(means, sigma)

# Determine if this percept was correct or incorrect
# A percept is correct if the sample is on the same side of the decision boundary as its mean
percepts_correct = (percepts > decision_boundary) == (means > decision_boundary)
percepts_incorrect = ~percepts_correct  # Incorrect percepts are the opposite of correct percepts

# Calculate the absolute distance of each mean from the decision boundary
distance_from_boundary = np.abs(means - decision_boundary)

# Define bins for sigma (standard deviation) and distance from boundary
sigma_bins = np.linspace(0.01, 4, 30)
distance_bins = np.linspace(0, 3, 30)

# Digitize the sigma and distance from boundary into bins
sigma_digitized = np.digitize(sigma, sigma_bins)
distance_digitized = np.digitize(distance_from_boundary, distance_bins)

# Create an empty 2D matrix to store confidence values
confidence_correct = np.zeros((len(distance_bins), len(sigma_bins)))
confidence_incorrect = np.zeros((len(distance_bins), len(sigma_bins)))

# Bias to adjust the confidence level at neutral evidence to 0.75
neutral_confidence = 0.75

# Calculate p(correct), confidence, for each bin
for i in range(1, len(distance_bins)):
    # bin_indices boolean array will only be 'True' when both conditions are met
    for j in range(1, len(sigma_bins)):
        bin_indices = (distance_digitized == i) & (sigma_digitized == j)
        if np.sum(bin_indices) > 0:
            # Calculate p(correct) for the bin
            correct_prob = np.mean(percepts_correct[bin_indices])

            # Calculate confidence for correct and incorrect outcomes
            confidence_correct[i - 1, j - 1] = 0.5 + (neutral_confidence - 0.5) * (correct_prob / 0.5)
            confidence_incorrect[i - 1, j - 1] = 0.5 + (neutral_confidence - 0.5) * ((1 - correct_prob) / 0.5)
                # note: 0.25 is the range in which confidence can vary in this case
                # note: divide by 0.5 to normalize confidence value to range between 0 and 1 (0.5 is the midpoint)

# Plotting heatmaps
plt.figure(figsize=(18, 8))

# Heatmap for confidence in correct trials
plt.subplot(1, 2, 1)
plt.imshow(confidence_correct.T, origin='lower', aspect='auto', extent=[sigma_bins[0], sigma_bins[-1], distance_bins[0], distance_bins[-1]], cmap='Greens')
plt.colorbar(label='Confidence (Correct)')
plt.xlabel('Noise (SD)')
plt.ylabel('Discriminability (Distance from DB)')
plt.title('Confidence in Correct Trials as Function of Noise and Discriminability')

# Heatmap for confidence in incorrect trials
plt.subplot(1, 2, 2)
plt.imshow(confidence_incorrect.T, origin='lower', aspect='auto', extent=[sigma_bins[0], sigma_bins[-1], distance_bins[0], distance_bins[-1]], cmap='Reds')
plt.colorbar(label='Confidence (Incorrect Trials)')
plt.xlabel('Noise (SD)')
plt.ylabel('Discriminability (Mean distance from DB)')
plt.title('Confidence in Incorrect Trials as Function of Noise and Discriminability')

plt.tight_layout()
plt.show()