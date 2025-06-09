from continous_distribution import pareto_distribution

if __name__ == "__main__":
    import numpy as np
    from scipy.stats import pareto
    from matplotlib import pyplot as plt

    # Parameters for Pareto distribution
    beta = 1.0  # Scale parameter
    ks = [2.05, 2.5, 3, 4]  # Shape parameters

    #true mean and variance
    true_mean = [beta*k / (k - 1) for k in ks if k > 1]
    true_variance = [beta**2*k / ((k - 1)**2 * (k - 2)) for k in ks if k > 2]
    samples = 10000  # Number of samples to generate
    pareto_samples = [pareto_distribution(samples, beta, k) for k in ks]
    #calculate the mean and variance of the samples
    sample_means = [np.mean(sample) for sample in pareto_samples]
    sample_variances = [np.var(sample) for sample in pareto_samples]

    # Print the true and sample means and variances
    for k, true_m, true_v, sample_m, sample_v in zip(ks, true_mean, true_variance, sample_means, sample_variances):
        print(f"Pareto (k={k}):")
        print(f"  True Mean: {true_m}, Sample Mean: {sample_m}")
        print(f"  True Variance: {true_v}, Sample Variance: {sample_v}")