import numpy as np
from ex2_3.discrete_distribution import plot_histogram, chi_square_test
from scipy.stats import expon, norm, pareto

def exponential_distribution(samples, lamda=1.0):
    U = np.random.uniform(0, 1, samples)
    return -np.log(U) / lamda

def normal_distribution_box_mueller(samples, mu=0.0, sigma=1.0):
    U1 = np.random.uniform(0, 1, samples)
    U2 = np.random.uniform(0, 1, samples)
    Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    Z1 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
    return mu + sigma * Z0, mu + sigma * Z1

def pareto_distribution(samples, beta=1.0, k=2.0):
    U = np.random.uniform(0, 1, samples)
    return beta * (U**(-1/k))

def run_chi_square_test(samples, analytical_dist, num_bins=50, range=None):
    counts, bin_edges = np.histogram(samples, bins=num_bins, range=range)
    
    bin_probs = analytical_dist.cdf(bin_edges[1:]) - analytical_dist.cdf(bin_edges[:-1])
    expected = bin_probs * len(samples)

    # Remove bins with expected < 5
    mask = expected >= 5
    counts = counts[mask]
    expected = expected[mask]
    num_bins = len(counts)

    return chi_square_test(counts, expected, num_bins, verbose=True)

if __name__ == "__main__":
    samples = 10000
    lamda = 1.0
    mu = 0.0
    sigma = 1.0
    beta = 1.0
    ks = [2.05, 2.5, 3, 4]

    # Generate Samples
    exp_samples = exponential_distribution(samples, lamda)
    norm_samples_0, norm_samples_1 = normal_distribution_box_mueller(samples, mu, sigma)
    pareto_samples = [pareto_distribution(samples, beta, k) for k in ks]

    # Plot and Test: Exponential
    plot_histogram(
        exp_samples,
        bins=50,
        title='Exponential Distribution',
        analytical_pdf=lambda x: expon.pdf(x, scale=1/lamda),
        pdf_range=(0, np.max(exp_samples)),
        save_figure=True,
        filename='exponential_distribution_histogram.png'
    )
    print("Chi-Square Test: Exponential")
    run_chi_square_test(exp_samples, expon(scale=1/lamda), num_bins=50)

    # Plot and Test: Normal (Z0)
    plot_histogram(
        norm_samples_0,
        bins=50,
        title='Normal Distribution (Z0)',
        analytical_pdf=lambda x: norm.pdf(x, loc=mu, scale=sigma),
        pdf_range=(mu - 4*sigma, mu + 4*sigma),
        save_figure=True,
        filename='normal_distribution_z0_histogram.png'
    )
    print("Chi-Square Test: Normal (Z0)")
    run_chi_square_test(norm_samples_0, norm(loc=mu, scale=sigma), num_bins=50)

    # Plot and Test: Pareto for each k
    for pareto_sample, k in zip(pareto_samples, ks):
        print(f"Chi-Square Test: Pareto (k={k})")
        plot_histogram(
            pareto_sample,
            bins=50,
            title=f'Pareto Distribution (k={k})',
            analytical_pdf=lambda x, k=k: pareto.pdf(x, b=k, scale=beta),
            pdf_range=(beta, np.percentile(pareto_sample, 99)),
            save_figure=True,
            filename=f'pareto_distribution_k_{k}_histogram.png'
        )
        run_chi_square_test(
            pareto_sample,
            pareto(b=k, scale=beta),
            num_bins=50,
            range=(beta, np.percentile(pareto_sample, 99))
        )
