import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_trials = 10000
sigma = 1.0  # Standard deviation (perceptual noise)
decision_boundary = 0  # Decision boundary (b)

# Generate means in the range [0, 4]. Means will always be on the right side but their distance from DB will vary.
means = np.random.uniform(0, 4, num_trials)

# Generate percepts for each mean
percepts = np.random.normal(means, sigma)

# Determine if each percept is correct (i.e., if they are on same side as their mean)
percepts_correct = (percepts > decision_boundary).astype(int)
    # astype(int) to convert 'True' to 1 and 'False' to 0

# Define bins for the full x-axis range
bins = np.linspace(-4, 4, 25)

# Function to calculate average accuracy from bin counts for both confidence levels
def calculate_average_accuracy(percepts, percepts_correct, means, bins):
    # Create arrays to store average accuracy for each bin
    low_conf_accuracies = np.full(len(bins) - 1, np.nan)
        # len(bins) - 1 gives # of elements in array
        # np.nan will start the array w/ Not A Number values
    high_conf_accuracies = np.full(len(bins) - 1, np.nan)

    for i in range(len(bins) - 1):
        bin_mask = (percepts >= bins[i]) & (percepts < bins[i + 1])

        # Compute discriminability for percepts in the bin
        bin_percepts = percepts[bin_mask]
        bin_percepts_correct = percepts_correct[bin_mask]
        bin_means = means[bin_mask]

        # Low-confidence trials
        low_conf_mask = (np.abs(bin_means - decision_boundary) <= 0.5)
        if np.any(low_conf_mask):
            # checks if any percepts in current bin meet low_bin_mask criteria, if yes then....
            low_conf_accuracies[i] = np.mean(bin_percepts_correct[low_conf_mask])
            # filter percepts_correct array to include only percepts that fit within low_bin_mask criteria and calulate their mean

        # High-confidence trials
        high_conf_mask = (np.abs(bin_means - decision_boundary) > 0.5)
        if np.any(high_conf_mask):
            # checks if any percepts in current bin meet high_bin_mask criteria, if yes then...
            high_conf_accuracies[i] = np.mean(bin_percepts_correct[high_conf_mask])
            # filters percepts_correct array to include only percepts that fit within high_bin_mask criteria and calulates their mean

    return low_conf_accuracies, high_conf_accuracies

# Compute average accuracies for both confidence levels
low_conf_accuracies, high_conf_accuracies = calculate_average_accuracy(percepts, percepts_correct, means, bins)



print(f"\nBin {i + 1}: {bins[i]:.2f} to {bins[i + 1]:.2f}")
print("  Low-confidence percepts and their accuracy:")
for percept, accuracy in zip(low_conf_percepts, low_conf_accuracies):
    print(f"    Percept: {percept:.2f}, Accuracy: {accuracy:.2f}")


# Call the function
print_low_confidence_percepts(percepts, percepts_correct, means, bins, decision_boundary)

# Compute bin centers
bin_centers = (bins[:-1] + bins[1:]) / 2

# Plot psychometric curves
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, low_conf_accuracies, label='Low confidence', color='red', marker='o', linestyle='-')
plt.plot(bin_centers, high_conf_accuracies, label='High confidence', color='blue', marker='o', linestyle='-')
plt.xlabel('Discriminability')
plt.ylabel('P(Choosing Right)')
plt.title('Conditioned Psychometric Curve')
plt.legend()
plt.grid(True)
plt.axhline(0.5, color='grey', linestyle='--')
plt.show()




