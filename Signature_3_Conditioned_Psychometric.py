import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_trials = 10000
sigma = 1.0  # Standard deviation (perceptual noise)
decision_boundary = 0  # Decision boundary (b)

# Randomly sample 10000 means (trials) which all differ in how far they are to the decision boundary
means = np.linspace(-3, 3, num_trials)

# Generate a random percept for each of these means
percepts = np.random.normal(means, sigma)

# Calculate distance of each mean from the decision boundary
distance_from_boundary = means - decision_boundary
distance_from_db = percepts - decision_boundary

# Define bins for the full x-axis range
bins = np.linspace(-3, 3, 25)

# Define confidence masks
low_conf_mask = (np.abs(percepts - decision_boundary) < 0.5)
med_conf_mask = (np.abs(percepts - decision_boundary) >= 0.5) & (np.abs(percepts - decision_boundary) < 1.1)
high_conf_mask = (np.abs(percepts - decision_boundary) >= 1.1)

# Function to calculate SEM
def calculate_sem(data):
    return np.std(data, ddof=1) / np.sqrt(len(data))
    # ddof = 1 divides n-1 as this is a sample and not entire population

# Initialize lists to store results
bin_centers = []
prob_choosing_right_low_conf = []
prob_choosing_right_med_conf = []
prob_choosing_right_high_conf = []
sem_low_conf = []
sem_med_conf = []
sem_high_conf = []

# Calculate P(choosing right)
def calculate_prob_choosing_right_side(percepts, distance_from_boundary, distance_from_db, bins):
    prob_choosing_right_low_conf = []
    prob_choosing_right_med_conf = []
    prob_choosing_right_high_conf = []
    sem_low_conf = []
    sem_med_conf = []
    sem_high_conf = []

    for i in range(len(bins) - 1):
        # Define the bin range
        bin_mask = (distance_from_boundary >= bins[i]) & (distance_from_boundary < bins[i + 1])

        # Calculate bin center for the point on the figure
        bin_center = (bins[i] + bins[i + 1]) / 2
        bin_centers.append(bin_center)

        # Low confidence trials
        low_conf_bin_mask = bin_mask & low_conf_mask
        if np.any(low_conf_bin_mask):
            prob_choosing_right_low_conf.append(np.mean(distance_from_db[low_conf_bin_mask] > 0))
            sem_low_conf.append(calculate_sem(distance_from_db[low_conf_bin_mask] > 0))
        else:
            prob_choosing_right_low_conf.append(0)
            sem_low_conf.append(0)

        # Medium confidence trials
        med_conf_bin_mask = bin_mask & med_conf_mask
        if np.any(med_conf_bin_mask):
            prob_choosing_right_med_conf.append(np.mean(distance_from_db[med_conf_bin_mask] > 0))
            sem_med_conf.append(calculate_sem(distance_from_db[med_conf_bin_mask] > 0))
        else:
            prob_choosing_right_med_conf.append(0)
            sem_med_conf.append(0)

        # High confidence trials
        high_conf_bin_mask = bin_mask & high_conf_mask
        if np.any(high_conf_bin_mask):
            prob_choosing_right_high_conf.append(np.mean(distance_from_db[high_conf_bin_mask] > 0))
            sem_high_conf.append(calculate_sem(distance_from_db[high_conf_bin_mask] > 0))
        else:
            prob_choosing_right_high_conf.append(0)
            sem_high_conf.append(0)

    return (prob_choosing_right_low_conf, sem_low_conf,
            prob_choosing_right_med_conf, sem_med_conf,
            prob_choosing_right_high_conf, sem_high_conf)

# Compute probabilities
(prob_choosing_right_low_conf, sem_low_conf,
 prob_choosing_right_med_conf, sem_med_conf,
 prob_choosing_right_high_conf, sem_high_conf) = calculate_prob_choosing_right_side(percepts, distance_from_boundary, distance_from_db, bins)

# Compute bin centers
bin_centers = (bins[:-1] + bins[1:]) / 2

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(bin_centers, prob_choosing_right_low_conf, yerr=sem_low_conf, label='Low conf', color='blue', marker='o', linestyle='-')
plt.errorbar(bin_centers, prob_choosing_right_med_conf, yerr=sem_med_conf, label='Med conf', color='green', marker='o', linestyle='-')
plt.errorbar(bin_centers, prob_choosing_right_high_conf, yerr=sem_high_conf, label='High conf', color='red', marker='o', linestyle='-')
plt.xlabel('Discriminability')
plt.ylabel('Choosing Right')
plt.title('Conditioned Psychometric')
plt.legend()
plt.grid(True)
plt.axhline(0.5, color='grey', linestyle='--')
plt.show()




