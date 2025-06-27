import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

def LCG(a=3, c=1, m=16, n=10):
    x = np.zeros(n, dtype=int)
    x[0] = 1  # Seed value

    for i in range(1, n):
        x[i] = (a * x[i - 1] + c) % m

    return x

def chi_square_test(observed, expected, num_bins, verbose=False):
    if len(observed) != len(expected):
        raise ValueError("Observed and expected arrays must have the same length.")
    T = np.sum((observed - expected) ** 2 / expected)
    chi_square_critical = chi2.sf(T, df=num_bins-1)
    if verbose:
        print(f"Chi-square statistic: {T}, Critical value: {chi_square_critical}")
    return chi_square_critical

def kolmogorov_smirnov_test(data1, data2, verbose=False):
    if len(data1) != len(data2):
        raise ValueError("Data arrays must have the same length.")
    d = np.max(np.abs(np.sort(data1) - np.sort(data2)))
    if verbose:
        print(f"Kolmogorov-Smirnov statistic: {d}")
    return d

def run_test(data, verbose=False): #1 in Slide
    median = np.median(data)
    runs_above = 0
    runs_below = 0
    prev = data[0]
    for idx, d in enumerate(data[1:]):
        if d > median:
            if prev < median:
                runs_above += 1
        elif d < median:
            if prev > median:
                runs_below += 1
        if idx == len(data) - 2:
            if d > median:
                runs_above += 1
            elif d < median:
                runs_below += 1
        prev = d
    if verbose:
        print(f"Median: {median}")
        print(f"Runs above median: {runs_above}, Runs below median: {runs_below}")
    return runs_above, runs_below, median

def correlation_coefficient(data, h=1, verbose=False):
    c_h = np.sum((data[:-h] - np.mean(data[:-h])) * (data[h:] - np.mean(data[h:]))) / (len(data) - h)
    if verbose:
        print(f"Correlation coefficient with lag {h}: {c_h}")
    return c_h

def histogram(data, bins=10):
    hist, bin_edges = np.histogram(data, bins=bins)
    return hist, bin_edges

if __name__ == "__main__":
    n = 10000
    a = 5
    c = 1
    m = 20
    num_bins = 16
    random_numbers = LCG(a, c, m, n)
    U = random_numbers / m
    print("Generated random numbers:", random_numbers)
    print("Mean:", np.mean(random_numbers))
    print("Variance:", np.var(random_numbers))
    print("Standard Deviation:", np.std(random_numbers))

    #test
    expected = np.full(num_bins, n / num_bins)
    observed, _ = histogram(U, bins=num_bins)
    
    chi_square = chi_square_test(observed, expected, num_bins=num_bins, verbose=True)
    d = kolmogorov_smirnov_test(U, np.linspace(0,1,n), verbose=True)
    runs_above, runs_below, median = run_test(U, verbose=True)
    c_h = correlation_coefficient(U, verbose=True, h=len(U) // 2)

    if True:  # Set to True to enable plotting
      #scatter plot U_i+1 vs U_i
      plt.figure(figsize=(10, 6))
      plt.scatter(U[:-1], U[1:], alpha=0.5)
      plt.xlabel("U_i")
      plt.ylabel("U_{i+1}")
      plt.grid()
      plt.title("Scatter Plot of Generated Random Numbers")
      plt.savefig("scatter_plot_bad.png")
      

      #plot
      hist, bin_edges = histogram(random_numbers, bins=num_bins)
      plt.hist(random_numbers, bins=bin_edges, edgecolor='black')
      plt.savefig("histogram_bad.png")