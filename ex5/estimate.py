import numpy as np
from scipy.stats import norm

def monte_carlo_estimator(samples = 100):
  U = np.random.uniform(0, 1, samples)
  return np.mean(np.exp(U))

def antithetic_variables_estimator(samples = 100):
  U = np.random.uniform(0, 1, samples // 2)
  U = np.concatenate((U, 1 - U))
  return np.mean(np.exp(U))

def control_variable_estimator(samples):
    U = np.random.uniform(0, 1, samples)
    # Y = np.exp(U)
    Y = -np.log(U)
    c = -np.cov(Y, U)[0, 1] / np.var(U)
    breakpoint()
    return np.mean(Y + c * (U - 0.5))

def stratified_sampling_estimator(samples = 100):
  strata = np.linspace(0, 1, num=10)
  estimates = []
  for i in range(len(strata) - 1):
    U = np.random.uniform(strata[i], strata[i + 1], samples // 10)
    estimates.append(np.mean(np.exp(U)))
  return np.mean(estimates)

if __name__ == "__main__":
    samples = 100  # Number of samples per estimate
    n_repeats = 1000  # Number of repetitions to compute CI
    confidence_level = 0.95
    z_score = norm.ppf((1 + confidence_level) / 2)  # 1.96 for 95% CI

    # Arrays to store estimates from multiple runs
    mc_estimates = np.zeros(n_repeats)
    antithetic_estimates = np.zeros(n_repeats)
    control_estimates = np.zeros(n_repeats)
    stratified_estimates = np.zeros(n_repeats)

    # Run each estimator multiple times
    for i in range(n_repeats):
        mc_estimates[i] = monte_carlo_estimator(samples)
        antithetic_estimates[i] = antithetic_variables_estimator(samples)
        control_estimates[i] = control_variable_estimator(samples)
        stratified_estimates[i] = stratified_sampling_estimator(samples)

    # Compute mean and standard error for each estimator
    estimators = [
        ("Monte Carlo", mc_estimates),
        ("Antithetic Variables", antithetic_estimates),
        ("Control Variable", control_estimates),
        ("Stratified Sampling", stratified_estimates)
    ]

    print(f"True value of E[e^U]: {np.exp(1) - 1:.6f}")
    print(f"Confidence Intervals (95%) with {samples} samples per estimate, {n_repeats} repetitions:\n")

    for name, estimates in estimators:
        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates, ddof=1)
        se = std_estimate / np.sqrt(n_repeats)  # Standard error
        ci_lower = mean_estimate - z_score * se
        ci_upper = mean_estimate + z_score * se
        print(f"{name} Estimate:")
        print(f"  Mean: {mean_estimate:.6f}")
        print(f"  Standard Deviation: {std_estimate:.6f}")
        print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]\n")