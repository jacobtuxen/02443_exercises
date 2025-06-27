import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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


if __name__ == "__main__":
    # Parameters
    TRUE_VALUE = np.exp(1) - 1
    THEORETICAL_LAMBDA = 1.3548
    num_iterations = 100
    sample_size = 1000
    lambda_values = np.linspace(0.1, 2.5, 100)

    lambdas = []
    all_variances = []

    # Main simulation loop
    for i in range(num_iterations):
        variances = []
        for lam in lambda_values:
            _, var = importance_sampling(sample_size=sample_size, lambda_val=lam)
            variances.append(var)
        
        all_variances.append(variances)

        # Find lambda with minimum variance in this iteration
        min_variance_index = np.argmin(variances)
        simulated_optimal_lambda = lambda_values[min_variance_index]
        lambdas.append(simulated_optimal_lambda)

    # Convert results to numpy arrays
    lambdas = np.array(lambdas)
    all_variances = np.array(all_variances)  # Shape: (num_iterations, len(lambda_values))

    # Compute mean and confidence interval of simulated optimal lambdas
    mean_lambda = np.mean(lambdas)
    std_lambda = np.std(lambdas, ddof=1)
    conf_int = stats.t.interval(0.95, df=num_iterations - 1, loc=mean_lambda, scale=std_lambda / np.sqrt(num_iterations))

    print(f"Mean of simulated optimal lambdas: {mean_lambda:.4f}")
    print(f"95% Confidence Interval: ({conf_int[0]:.4f}, {conf_int[1]:.4f})")

    # Histogram of simulated optimal lambdas
    plt.figure(figsize=(8, 5))
    plt.hist(lambdas, bins=15, edgecolor='black', alpha=0.7)
    plt.axvline(mean_lambda, color='blue', linestyle='--', label=f'Mean λ = {mean_lambda:.3f}')
    plt.axvline(THEORETICAL_LAMBDA, color='red', linestyle=':', label='Theoretical λ = 1.3548')
    plt.title("Distribution of Simulated Optimal λ")
    plt.xlabel("Lambda")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig('ex5/figures/lambda_distribution.png', dpi=300)
    plt.show()

    # Mean variance curve
    mean_variance_curve = np.mean(all_variances, axis=0)
    lambda_min_mean_variance_index = np.argmin(mean_variance_curve)
    lambda_min_mean_variance = lambda_values[lambda_min_mean_variance_index]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, mean_variance_curve, label='Mean IS Variance Across Iterations', color='blue')
    plt.axvline(x=mean_lambda, color='green', linestyle='--', label=f'Mean Optimal λ ≈ {mean_lambda:.3f}')
    plt.axvline(x=THEORETICAL_LAMBDA, color='red', linestyle=':', label='Theoretical λ = 1.3548')
    plt.axvline(x=lambda_min_mean_variance, color='purple', linestyle='-.', label=f'Min Mean-Variance λ ≈ {lambda_min_mean_variance:.3f}')
    plt.title('Mean Variance of Importance Sampling vs. λ (across simulations)')
    plt.xlabel('λ (lambda)')
    plt.ylabel('Mean Variance')
    plt.legend()
    plt.ylim(0, min(20, max(mean_variance_curve)))  # Limit y-axis for visibility
    plt.savefig('ex5/figures/IS_mean_variance_curve.png', dpi=300, bbox_inches='tight')
    plt.show()