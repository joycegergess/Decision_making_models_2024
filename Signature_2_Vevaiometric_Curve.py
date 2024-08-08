import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_trials = 10000
sigma = 1.0  # Standard deviation (perceptual noise remains same for all trials)
decision_boundary = 0  # Decision boundary (b)

# Randomly sample 10000 means (trials) which all differ in how far they are to the decision boundary
means = np.linspace(-4, 4, num_trials)

# Generate a random percept for each of these means and associate them with an outcome
samples = np.random.normal(means, sigma)

# Determine whether the percept is correct
# A percept is correct if the sample is on the same side of the decision boundary as its mean
percepts_correct = ((samples > decision_boundary) & (means > decision_boundary)) | ((samples < decision_boundary) & (means < decision_boundary))
percepts_incorrect = ~percepts_correct  # Incorrect percepts are the opposite of correct percepts

# Calculate absolute distance of each mean from decision-boundary, this will be my x-axis, how discriminable is the given stimulus
distance_from_boundary = np.abs(means - decision_boundary)


# Define the bin edges for the 20-point bins
bin_edges = np.linspace(0, 4, 21)

# Arrays to store bin centers and confidence values for correct and incorrect outcomes
bin_centers = []
confidence_values_correct = []

confidence_values_incorrect = []

# Bias to adjust the confidence level at neutral evidence to 0.75
neutral_confidence = 0.75

# Calculate confidence for each bin
for i in range(len(bin_edges) - 1):
    # Find indices of percepts that fall into the current bin
    bin_indices = np.where((distance_from_boundary >= bin_edges[i]) & (distance_from_boundary < bin_edges[i + 1]))[0]
    # np.where returns percepts from each bin depending on their distance from DB
    # bin_edges[i + 1] represents upper boundary of current bin
    # purpose of [0] here is to get the actual array of indices from the tuple returned by np.where

    # Get samples and outcomes for the current bin using indices
    bin_percepts_correct = percepts_correct[bin_indices]
    bin_percepts_incorrect = percepts_incorrect[bin_indices]

    # Calculate the center of the bins to know where to plot the points on the x-axis
    bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
    bin_centers.append(bin_center)

    # Calculate the confidence for correct and incorrect outcomes
    if len(bin_percepts_correct) > 0:
        confidence_correct = np.mean(bin_percepts_correct)
        confidence_incorrect = np.mean(bin_percepts_incorrect)
        confidence_correct = 0.5 + (neutral_confidence - 0.5) * (confidence_correct / 0.5)
        confidence_incorrect = 0.5 + (neutral_confidence - 0.5) * (confidence_incorrect / 0.5)

    else:
        confidence_correct = 0.5
        confidence_incorrect = 0.5

    confidence_values_correct.append(confidence_correct)
    confidence_values_incorrect.append(confidence_incorrect)
    # this appends (adds) the calculated midpoints (bin_center) to the bin_centers array created above

# Convert lists to arrays
bin_centers = np.array(bin_centers)
confidence_values_correct = np.array(confidence_values_correct)
confidence_values_incorrect = np.array(confidence_values_incorrect)

# Plotting
plt.figure(figsize=(10, 10))
plt.plot(bin_centers, confidence_values_correct, 'g-', label='Correct')
plt.plot(bin_centers, confidence_values_incorrect, 'r-', label='Incorrect')
plt.axhline(y=0.75, color='k', linestyle='--')  # Line at y=0.75
plt.xlabel('Discriminability')
plt.ylabel('Confidence')
plt.ylim(0.5, 1)
plt.title('Vevaiometric Curve')
plt.legend()
plt.show()


