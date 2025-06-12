import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def crude_monte_carlo(a, N):
    samples = np.random.randn(N)
    estimate = np.mean(samples > a)
    std_error = np.std(samples > a) / np.sqrt(N)
    return estimate, std_error

def importance_sampling(a, sigma2, N):
    sigma = np.sqrt(sigma2)
    samples = np.random.normal(loc=a, scale=sigma, size=N)
    
    # Compute weights
    p = norm.pdf(samples, loc=0, scale=1)
    q = norm.pdf(samples, loc=a, scale=sigma)
    
    weights = p / q
    indicators = samples > a
    estimate = np.mean(indicators * weights)
    std_error = np.std(indicators * weights) / np.sqrt(N)
    
    return estimate, std_error

# Parameters
a_values = [2, 4]
sample_sizes = [10**3, 10**4, 10**5]
sigma2 = 1

# Run experiments
for a in a_values:
    print(f"\nEstimating P(Z > {a})")
    true_val = 1 - norm.cdf(a)
    print(f"True value: {true_val:.6f}")
    
    for N in sample_sizes:
        crude_est, crude_err = crude_monte_carlo(a, N)
        is_est, is_err = importance_sampling(a, sigma2, N)
        
        print(f"  N = {N}")
        print(f"    Crude MC: estimate = {crude_est:.6f}, error ≈ {crude_err:.6f}")
        print(f"    Importance Sampling: estimate = {is_est:.6f}, error ≈ {is_err:.6f}")
