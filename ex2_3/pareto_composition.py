from ex2_3.discrete_distribution import crude_method, plot_histogram
import numpy as np

def pareto_composition(samples=10000, beta=1.0, ks=[3,4], probabilities=[0.5, 0.5]):
    if type(ks) is list:
        ks = np.array(ks)
    U = np.random.uniform(0, 1, samples)
    p_i = crude_method(probabilities, size=samples) - 1
    return beta * (U**(-1/ks[p_i]))

def exponential_composition(samples=10000, lambdas=[1.0, 2.0], probabilities=[0.5, 0.5]):
    if type(lambdas) is list:
        lambdas = np.array(lambdas)
    U = np.random.uniform(0, 1, samples)
    p_i = crude_method(probabilities, size=samples) - 1
    return -np.log(U) / lambdas[p_i]

if __name__ == "__main__":
    probabilities = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
    ks = [3, 4, 5, 6, 7 ,8]
    samples = 100000
    beta = 1.0
    pareto_samples = pareto_composition(samples, beta, ks, probabilities)

    plot_histogram(
        pareto_samples,
        bins=50,
        title='Pareto Composition Distribution',
        pdf_range=(0, np.max(pareto_samples)),
        save_figure=True,
        filename='pareto_composition_distribution_histogram.png'
    )

    