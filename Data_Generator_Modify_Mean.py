# Parameters
num_trials = 10000
sigma = 1.0  # Standard deviation (perceptual noise)
decision_boundary = 0  # Decision boundary (b)

# Randomly generate 1000 different means (trials) which all differ in how far they are to the decision boundary
means = np.random.uniform(-3, 3, num_trials)

# Generate a percept for each trial based on its mean and standard deviation
percepts = np.random.normal(means, sigma)

# Determine if this percept was correct or incorrect
# A percept is correct if the sample is on the same side of the decision boundary as its mean
percepts_correct = (percepts > decision_boundary) == (means > decision_boundary)
