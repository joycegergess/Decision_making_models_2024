import numpy as np
import matplotlib.pyplot as plt


# Function to calculate SEM
def calculate_sem(data):
    return np.std(data, ddof=1) / np.sqrt(len(data))

# Parameters
num_trials = 10000
sigma = 1.0  # Standard deviation (perceptual noise remains the same for all trials)
decision_boundary = 0  # Decision boundary (b)

# Signature 1: Calibration Curve

# Randomly generate different means for trials (for calibration curve)
calibration_means = np.random.uniform(-3, 3, num_trials)

# Generate samples for calibration curve
calibration_samples = np.random.normal(calibration_means, sigma)

# Determine whether the percept is correct (if percept falls on same as its mean)
calibration_percepts_correct = (calibration_samples > decision_boundary) == (calibration_means > decision_boundary)

# Calculate percepts' distance from the decision boundary
calibration_distance_from_boundary = np.abs(calibration_samples - decision_boundary)

# Define bin edges for calibration curve
bin_edges = np.linspace(0, 3, 21)

# Arrays to store calibration curve data
calibration_bin_centers = []
calibration_confidence_values = []
calibration_sem_values = []

# Calculate calibration curve
for i in range(len(bin_edges) - 1):
    # Find indices of percepts that fall into the current bin
    bin_indices = np.where((calibration_distance_from_boundary >= bin_edges[i]) &
                           (calibration_distance_from_boundary < bin_edges[i + 1]))[0]
    # Calculate center of the bins to know where to plot the points on x-axis in calibration curve
    bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
    calibration_bin_centers.append(bin_center)

    # Get correct percepts for the current bin using indices
    bin_calibration_percepts_correct = calibration_percepts_correct[bin_indices]

    # Calculate the mean and SEM of each bin
    if len(bin_calibration_percepts_correct) > 0:
        confidence = np.mean(bin_calibration_percepts_correct)
        sem = calculate_sem(bin_calibration_percepts_correct)
    else:
        confidence = 0
        sem = 0

    calibration_confidence_values.append(confidence)
    calibration_sem_values.append(sem)

calibration_bin_centers = np.array(calibration_bin_centers)
calibration_confidence_values = np.array(calibration_confidence_values)
calibration_sem_values = np.array(calibration_sem_values)

# Signature 2: Vevaiometric Curve

means = np.linspace(-3, 3, num_trials)
percepts = np.random.normal(means, sigma)

# Determine whether the percept is correct or incorrect
percepts_correct = (percepts > decision_boundary) == (means > decision_boundary)
percepts_incorrect = ~percepts_correct  # Incorrect percepts are the opposite of correct percepts

# Calculate MEANS' distance from the decision boundary
distance_from_boundary = np.abs(means - decision_boundary)

# Arrays to store vevaiometric curve data
vevaiometric_bin_centers = []
vevaiometric_confidence_correct = []
vevaiometric_confidence_incorrect = []
vevaiometric_sem_correct = []
vevaiometric_sem_incorrect = []

# Calculate vevaiometric curve using calibration data
for i in range(len(bin_edges) - 1):
    # Find indices of percepts that fall into the current bin
    bin_indices = np.where((distance_from_boundary >= bin_edges[i]) & (distance_from_boundary < bin_edges[i + 1]))[0]

    if len(bin_indices) > 0:
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        vevaiometric_bin_centers.append(bin_center)

        bin_percepts_correct = percepts_correct[bin_indices]
        bin_percepts_incorrect = percepts_incorrect[bin_indices]

        # Calculate confidence for correct percepts
        bin_confidences_correct = []
        for percept in percepts[bin_indices][bin_percepts_correct]:
            percept_distance = np.abs(percept - decision_boundary)
            closest_bin_index = np.argmin(np.abs(calibration_bin_centers - percept_distance))
            confidence = calibration_confidence_values[closest_bin_index]
            bin_confidences_correct.append(confidence)

        # Calculate confidence for incorrect percepts
        bin_confidences_incorrect = []
        for percept in percepts[bin_indices][bin_percepts_incorrect]:
            percept_distance = np.abs(percept - decision_boundary)
            closest_bin_index = np.argmin(np.abs(calibration_bin_centers - percept_distance))
            confidence = calibration_confidence_values[closest_bin_index]
            bin_confidences_incorrect.append(confidence)

        if len(bin_confidences_correct) > 0:
            mean_confidence_correct = np.mean(bin_confidences_correct)
            sem_correct = calculate_sem(bin_confidences_correct)
        else:
            mean_confidence_correct = 0
            sem_correct = 0

        if len(bin_confidences_incorrect) > 0:
            mean_confidence_incorrect = np.mean(bin_confidences_incorrect)
            sem_incorrect = calculate_sem(bin_confidences_incorrect)
        else:
            mean_confidence_incorrect = 0
            sem_incorrect = 0

        vevaiometric_confidence_correct.append(mean_confidence_correct)
        vevaiometric_confidence_incorrect.append(mean_confidence_incorrect)
        vevaiometric_sem_correct.append(sem_correct)
        vevaiometric_sem_incorrect.append(sem_incorrect)

vevaiometric_bin_centers = np.array(vevaiometric_bin_centers)
vevaiometric_confidence_correct = np.array(vevaiometric_confidence_correct)
vevaiometric_confidence_incorrect = np.array(vevaiometric_confidence_incorrect)
vevaiometric_sem_correct = np.array(vevaiometric_sem_correct)
vevaiometric_sem_incorrect = np.array(vevaiometric_sem_incorrect)

# Plot Calibration Curve
plt.figure(figsize=(10, 5))
plt.errorbar(calibration_bin_centers, calibration_confidence_values, yerr=calibration_sem_values, fmt='b-', capsize=5)
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.xlabel('Distance to DB')
plt.ylabel('Confidence')
plt.ylim(0, 1)
plt.title('Calibration Curve')
plt.show()

# Plot Vevaiometric Curve (Correct and Incorrect)
plt.figure(figsize=(10, 5))
plt.errorbar(vevaiometric_bin_centers, vevaiometric_confidence_correct, yerr=vevaiometric_sem_correct, fmt='g-',
             label='Correct', capsize=5)
plt.errorbar(vevaiometric_bin_centers, vevaiometric_confidence_incorrect, yerr=vevaiometric_sem_incorrect, fmt='r-',
             label='Incorrect', capsize=5)
plt.axhline(y=0.5, color='k', linestyle='--')
plt.xlabel('Discriminability (Mean Distance to DB)')
plt.ylabel('Confidence')
plt.ylim(0.5, 1)
plt.title('Vevaiometric Curve Using Calibration Curve')
plt.legend()
plt.show()

