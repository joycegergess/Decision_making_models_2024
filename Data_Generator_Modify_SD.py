import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# Parameters
num_trials = 10000
mean = 1.0  # Provided evidence
decision_boundary = 0  # Decision boundary


# Function to simulate percepts
def simulate_percepts(mean, sigma, num_trials):
    return np.random.normal(mean, sigma, num_trials)


# Function to compute the negative log-likelihood
def compute_nll(mean, sigma, decision_boundary, percepts):
    # Compute distances as a scalar
    distances = np.abs(mean - decision_boundary)

    # Compute probabilities of being correct (scalar value)
    probs_correct = norm.cdf(distances / sigma)

    # Determine correctness (boolean array)
    correctness = (percepts > decision_boundary) == (mean > decision_boundary)

    # Broadcast probs_correct to match the shape of correctness
    probs_correct_array = np.full_like(correctness, probs_correct)

    # Compute log probabilities
    log_probs = np.log(probs_correct_array[correctness] + 1e-10)  # Add small constant for numerical stability

    # Return negative log-likelihood
    return -np.sum(log_probs)


# Objective function for optimization
def nll_function(sigma):
    sigma = sigma[0]  # Extract scalar from array (minimize expects array input)
    percepts = simulate_percepts(mean, sigma, num_trials)
    return compute_nll(mean, sigma, decision_boundary, percepts)


# Initial guess for sigma
initial_sigma = [1.0]  # Minimize expects input as an array

# Optimization using minimize
result = minimize(
    nll_function,  # Objective function
    x0=initial_sigma,  # Initial guess
    bounds=[(0.01, 4)],  # Bounds for sigma
    method='L-BFGS-B'  # Optimization method
)

# Extract the optimal sigma
best_sigma = result.x[0]
print(f"Optimal sigma: {best_sigma}")

# Optional: Plot NLL as a function of sigma
sigmas = np.linspace(0.01, 4, 100)
nll_values = [nll_function([sigma]) for sigma in sigmas]

import matplotlib.pyplot as plt

plt.plot(sigmas, nll_values)
plt.xlabel('Sigma')
plt.ylabel('Negative Log-Likelihood (NLL)')
plt.title('NLL as a Function of Sigma')
plt.show()
