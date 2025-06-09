import numpy as np
import time
from scipy.stats import chi2

def geometric_distribution(p, size=1):
    if not (0 < p < 1):
        raise ValueError("Probability p must be in the range (0, 1).")
    
    return np.random.geometric(p, size)

def plot_histogram(data, bins=30, save_figure=False, filename='histogram.png', title=None, analytical_pdf=None, pdf_range=None):
    import matplotlib.pyplot as plt

    if analytical_pdf is not None and pdf_range is not None:
        x = np.linspace(*pdf_range, 1000)
        y = analytical_pdf(x)
        plt.plot(x, y, 'r-', label='Analytical PDF')
    
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='g')
    plt.title(title if title else 'Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(filename) if save_figure else None
    plt.show()

def crude_method(distribution, size=10000):
    data = []
    for i in range(size):
        random_value = np.random.uniform(0, 1)
        total_prob = 0
        prev_prob = 0
        for idx, prob in enumerate(distribution):
            total_prob += prob
            if random_value < total_prob and random_value > prev_prob:
                data.append(idx + 1)
                break
            prev_prob = total_prob
    data = np.array(data)
    
    return data

def rejection_method(distribution, size=10000, verbose=False):
    data = []
    num_guesses = 0
    max_prob = max(distribution)
    while len(data) < size:
        random_value = np.random.uniform(0, 1)
        num_guesses += 1
        I = np.floor(len(distribution) * random_value) + 1
        if np.random.uniform(0, 1) <= distribution[int(I) - 1] / max_prob:
            data.append(I)
    data = np.array(data)
    
    if verbose:
        print(f"Rejection method took {num_guesses} guesses to generate {size} samples.")
    return data

def alias_method(distribution, size=10000):
    n = len(distribution)
    distribution = np.array(distribution)
    scaled_distribution = distribution * n
    alias = np.zeros(n, dtype=int)
    prob = np.zeros(n)

    small = []
    large = []

    for i, prob_i in enumerate(scaled_distribution):
        if prob_i < 1:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        less = small.pop()
        more = large.pop()

        prob[less] = scaled_distribution[less]
        alias[less] = more

        scaled_distribution[more] = (scaled_distribution[more] + scaled_distribution[less]) - 1

        if scaled_distribution[more] < 1:
            small.append(more)
        else:
            large.append(more)

    for i in small:
        prob[i] = 1

    for i in large:
        prob[i] = 1

    data = np.zeros(size, dtype=int)
    for i in range(size):
        random_value = np.random.uniform(0, 1)
        index = int(random_value * n)
        if np.random.uniform(0, 1) < prob[index]:
            data[i] = index + 1
        else:
            data[i] = alias[index] + 1

    return data

def chi_square_test(observed, expected, num_bins, verbose=False):
    if len(observed) != len(expected):
        raise ValueError("Observed and expected arrays must have the same length.")
    T = np.sum((observed - expected) ** 2 / expected)
    chi_square_critical = chi2.sf(T, df=num_bins-1)
    if verbose:
        print(f"Chi-square statistic: {T}, Critical value: {chi_square_critical}")
    return chi_square_critical

if __name__ == "__main__":
    methods = ['Crude Method', 'Rejection Method', 'Alias Method']
    samples = 100000

    for method in methods:
      time_0 = time.time()
      distribution = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
      if method == 'Crude Method':
          data_from_dist = crude_method(distribution, size=samples)
      elif method == 'Rejection Method':
          data_from_dist = rejection_method(distribution, size=samples, verbose=True)
      elif method == 'Alias Method':
          data_from_dist = alias_method(distribution, size=samples)
      else:
          raise ValueError(f"Unknown method: {method}")
      time_1 = time.time()
      print(f"{method} took {time_1 - time_0:.4f} seconds to generate samples.")
      #test chi-square
      num_bins = len(distribution)
      expected = np.array(distribution) * samples
      observed, _ = np.histogram(data_from_dist, bins=np.arange(1, num_bins + 2) - 0.5)
      chi_square_critical = chi_square_test(observed, expected, num_bins, verbose=True)
    # plot_histogram(data_from_dist, bins=30, save_figure=True, filename='alias_method_histogram.png', method=f'{method}')