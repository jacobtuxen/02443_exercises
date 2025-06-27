import numpy as np
import math
from scipy.stats import chisquare
import matplotlib.pyplot as plt

RNG = np.random.rand(10000)

def plot_histogram(data, bins=6):
    plt.hist(data, bins=bins, density=True)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram of LCG Output')
    plt.grid(True)
    plt.show()

def geometric_dist(p, rng):
    X = []
    for u in rng:
        x = math.floor(math.log(1 - u) / math.log(1 - p)) + 1
        X.append(x)
    return np.array(X)

def gof_geometric(X, p, alpha=0.05):
    values, observed = np.unique(X, return_counts=True)

    expected_probs = [(1 - p) ** (k - 1) * p for k in values]
    expected = np.array(expected_probs) * len(X)

    while any(expected < 5):
        expected[-2] += expected[-1]
        observed[-2] += observed[-1]
        expected = expected[:-1]
        observed = observed[:-1]

    expected *= observed.sum() / expected.sum()

    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    print(f"Chi-square statistic: {chi2_stat:.4f}, p-value: {p_value:.4f}")


if __name__ == "__main__":
    p = 0.8
    X = geometric_dist(p, RNG)

    plot_histogram(X, bins = 30)

    gof_geometric(X, p)