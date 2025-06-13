# from estimate import monte_carlo_estimator
import numpy as np
import matplotlib.pyplot as plt

def h(x):
    return np.exp(x)

def f(x):
    return np.where((0 <= x) & (x <= 1), 1.0, 0.0)

def g(x, lam):
    return np.where(x >= 0, lam * np.exp(-lam * x), 0)

TRUE_VALUE = np.exp(1) - 1

def importance_sampling(lambda_val, sample_size=1000):
    Y = np.random.exponential(scale=1.0/lambda_val, size=sample_size)
    
    mask = (Y >= 0) & (Y <= 1)
    valid_Y = Y[mask]
    
    weights = np.zeros(sample_size)
    weights[mask] = h(valid_Y) * f(valid_Y) / g(valid_Y, lambda_val)
    
    integral_estimate = np.mean(weights)
    variance_estimate = np.var(weights, ddof=1) / sample_size  # Variance of the estimator
    
    return integral_estimate, variance_estimate


sample_size = 1000

lambda_values = np.linspace(0.1, 2.5, 100)
variances = []

for lam in lambda_values:
    _, var = importance_sampling(sample_size=sample_size, lambda_val=lam)
    variances.append(var)

min_variance_index = np.argmin(variances)
simulated_optimal_lambda = lambda_values[min_variance_index]
min_variance = variances[min_variance_index]

print(f"True value of the integral: {TRUE_VALUE:.5f}\n")
print(f"Theoretical optimal lambda ≈ 1.3548")
print(f"Simulated optimal lambda: {simulated_optimal_lambda:.3f}\n")
print(f"Minimum variance with Importance Sampling (at λ={simulated_optimal_lambda:.2f}): {min_variance:.5f}")


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, variances, label='Importance Sampling Variance')
plt.axvline(x=simulated_optimal_lambda, color='g', linestyle=':', label=f'Simulated Optimal λ ({simulated_optimal_lambda:.2f})')
plt.axvline(x=1.3548, color='r', linestyle=':', label=f'Calculated Optimal λ (1.3548)')
plt.title('Variance of Importance Sampling vs. λ')
plt.xlabel('λ (lambda)')
plt.ylabel('Variance')
plt.legend()
plt.ylim(0, min(20, max(variances))) # Cap y-axis for better visualization
plt.savefig('ex5/figures/IS_exercise_5_8.png', dpi=300, bbox_inches='tight')
plt.show()