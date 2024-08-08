# Take 2 for signature 3 (psychometric curve).
# This time we will not be measure the p(correct).
# Instead, we will measure p(choosing left) regardless of it is the correct choice or not.
# Goal of this plot is just to examine how choice behavior changes based on evidence.

import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_trials = 10000
sigma = 1.0  # Standard deviation (perceptual noise)
decision_boundary = 0  # Decision boundary (b)