import numpy as np

def bootstrap(data, n_iterations=1000, sample_size=None):
    if sample_size is None:
        sample_size = len(data)
    
    n_data = len(data)
    samples = np.empty((n_iterations, sample_size))
    
    for i in range(n_iterations):
        indices = np.random.randint(0, n_data, size=sample_size)
        samples[i] = data[indices]
    
    return samples

def pareto_distribution(samples=100, beta=1, k=1.05):
    return beta * (np.random.pareto(k, size=samples) + 1)

if __name__ == "__main__":
    N = 200
    data = pareto_distribution(N, beta=1.0, k=2.0)
    mean = np.mean(data)
    median = np.median(data)
    bootstrap_samples = bootstrap(data, n_iterations=100, sample_size=N)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    bootstrap_medians = np.median(bootstrap_samples, axis=1)

    import matplotlib.pyplot as plt
    plt.hist(bootstrap_means, bins=30, alpha=0.5, label='Bootstrap Means')
    plt.hist(bootstrap_medians, bins=30, alpha=0.5, label='Bootstrap Medians')
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(median, color='blue', linestyle='dashed', linewidth=1, label='Median')
    plt.title('Bootstrap Means and Medians')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('bootstrap_distribution.png')
    plt.show()