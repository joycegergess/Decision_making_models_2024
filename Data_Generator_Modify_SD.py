import numpy as np

# Parameters
num_trials = 10000
mean = 1.0  # Provided evidence (x)
decision_boundary = 0  # Decision boundary (b)

# Randomly generate 1000 different trials which all differ in how big the standard deviation is
sigma = np.random.uniform(0.01, 4, num_trials)

# Generate a percept for each trial based on its mean and standard deviation
percepts = np.random.normal(mean, sigma)

# Determine if this percept was correct or incorrect
# A percept is correct if the sample is on the same side of the decision boundary as its mean
percepts_correct = (percepts > decision_boundary) == (means > decision_boundary)