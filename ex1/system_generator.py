import numpy as np
from LCG import LCG, chi_square_test, kolmogorov_smirnov_test, run_test, correlation_coefficient, histogram
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    m = 2**32-1
    random_numbers = np.random.randint(0, m, size=10000)  # Simulating LCG output
    U = random_numbers / m
    num_bins = 16
    n = len(random_numbers)
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
      plt.savefig("scatter_plot_np.png")
      

      #plot
      hist, bin_edges = histogram(random_numbers, bins=num_bins)
      plt.hist(random_numbers, bins=bin_edges, edgecolor='black')
      plt.savefig("histogram_np.png")