import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_trials = 10000
sigma = 1.0  # Standard deviation (perceptual noise remains same for all trials)
decision_boundary = 0  # Decision boundary (b)

# Randomly generate 1000 different means (trials) which all differ in how far they are to the decision boundary
means = np.random.uniform(-3, 3, num_trials)

# Generate a sample for each trial based on its mean
samples = np.random.normal(means, sigma)

# Determine whether the percept is correct
# A percept is correct if the sample is on the same side of the decision boundary as its mean
percepts_correct = (samples > decision_boundary) == (means > decision_boundary)
percepts_incorrect = ~percepts_correct  # Incorrect percepts are the opposite of correct percepts

# Calculate distance of each PERCEPT from decision-boundary, this will be my x-axis
distance_from_boundary = samples - decision_boundary

# Define bins and calculate the empirical probability of correct percepts
bins = np.linspace(-3, 3, 100)
bin_centers = []
prob_correct = []

for i in range(len(bins) - 1):
    bin_mask = (samples >= bins[i]) & (samples < bins[i + 1])
    prob_correct.append(percepts_correct[bin_mask].mean())

    # Calculate the center of the current bin
    bin_center = (bins[i] + bins[i + 1]) / 2
    bin_centers.append(bin_center)

# Convert lists to numpy arrays for plotting
bin_centers = np.array(bin_centers)
prob_correct = np.array(prob_correct)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the blue linked points for empirical P(reward)
plt.plot(bin_centers, prob_correct, color='blue', label='P(reward)', marker='o')

# Scatter plot with green ticks for correct percepts and red ticks for incorrect percepts
plt.scatter(samples[percepts_correct], np.zeros_like(samples[percepts_correct]), color='green', marker='|', s=100, label='Correct Percepts')
plt.scatter(samples[percepts_incorrect], np.zeros_like(samples[percepts_incorrect]), color='red', marker='|', s=100, label='Incorrect Percepts')

# Plot settings
plt.axhline(y=0.5, color='gray', linestyle='--')  # Plot a horizontal line at y=0.5 for reference
plt.axvline(x=0, color='black', linestyle='-')  # Plot a vertical line at the decision boundary for reference
plt.xlabel('Decision Axis')
plt.ylabel('P(reward)')
plt.xlim(-3, 3)
plt.ylim(0, 1)
plt.title('Probability of Getting a Reward vs. Distance of Percept to Decision Boundary')
plt.legend()
plt.show()
