import numpy as np
import math
from scipy.stats import chisquare

def random_walk_proposal(current_state):
    """Generates a new state by adding a small random step to the current state."""
    step_size = 0.5  # Define the step size for the random walk
    return current_state + np.random.normal(0, step_size)

def discrete_walk_proposal(current_state):
    possible_steps = [-1 ,0 ,1]  # Define possible steps
    return current_state + np.random.choice(possible_steps)

def metropolis_hastings(target_dist, proposal_dist, initial_state, num_samples, use_joint=False):
    if use_joint:
        samples = np.zeros((num_samples, 2))  # For joint distribution
    else:
        samples = np.zeros(num_samples)
    current_state = initial_state

    for i in range(num_samples):
        proposed_state = proposal_dist(current_state)
        acceptance_ratio = target_dist(proposed_state) / (target_dist(current_state) + 1e-10)  # Avoid division by zero

        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state
        
        samples[i] = current_state

    return samples

def truncated_poisson(c, A, i):
    i = int(i)  # Ensure i is an integer
    if i < 0:
        return 0.0  # Return 0 for negative indices
    return c * A**i / math.factorial(i)

def target_distribution_joint(c, A, i, j):
    m = 10 # hardcorde for now
    i = int(i)  # Ensure i is an integer
    j = int(j)  # Ensure i is an integer
    if i < 0 or j < 0 or i > m or j > m:  
        return 0.0  # Return 0 for negative indices
    return c * A[0]**i / math.factorial(i) * c * A[1]**j / math.factorial(j)
    
if __name__ == "__main__":
    # Example parameters
    A = 2.0
    c = 1
    
    num_samples = 30000
    burn_in = int(0.3 * num_samples)
    initial_state = 0
    target_dist = lambda i: truncated_poisson(c, A, i)
    samples = metropolis_hastings(target_dist, discrete_walk_proposal, initial_state, num_samples)
    samples = samples[burn_in:]
    #test with chi-squared test
    
    observed_counts, _ = np.histogram(samples, bins=np.arange(0, max(samples)))
    
    max_k = int(samples.max())
    target_probs = np.array([truncated_poisson(c, A, i) for i in range(max_k-1)])
    target_probs /= target_probs.sum()
    expected_counts = target_probs * observed_counts.sum()
    chi2_stat, p_value = chisquare(observed_counts, expected_counts)
    print(f"Chi-squared statistic: {chi2_stat}, p-value: {p_value}")
    print(f"Generated {num_samples} samples from the target distribution.")

    #histogram of samples
    import matplotlib.pyplot as plt
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')
    plt.title(f'MCMC Samples from Truncated Possion (c={c}, A={A})')
    plt.xlabel('Sample Value')
    plt.ylabel('Density')
    plt.savefig('mcmc_truncated_poisson.png')
    plt.show()