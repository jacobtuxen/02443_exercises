import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

# === Step (a): Generate (theta, psi) from the prior ===
def sample_from_prior(rho=0.5):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    xi, gamma = np.random.multivariate_normal(mean, cov)
    theta = np.exp(xi)
    psi = np.exp(gamma)
    return theta, psi

# === Step (b): Generate data X_i ~ N(theta, psi) ===
def generate_data(theta, psi, n):
    return np.random.normal(loc=theta, scale=np.sqrt(psi), size=n)

# === Step (c): Define posterior up to proportionality ===
def log_joint_posterior(theta, psi, data, rho=0.5):
    if theta <= 0 or psi <= 0:
        return -np.inf
    
    n = len(data)
    sample_mean = np.mean(data)
    sample_var = np.var(data, ddof=1)

    # Log-likelihood: X_i ~ N(theta, psi)
    ll = -0.5 * n * np.log(2 * np.pi * psi) - (1 / (2 * psi)) * np.sum((data - theta)**2)

    # Log-prior from f(theta, psi):
    xi = np.log(theta)
    gamma = np.log(psi)
    quad = (xi**2 - 2*rho*xi*gamma + gamma**2) / (2*(1 - rho**2))
    log_prior = -np.log(theta * psi * 2 * np.pi * np.sqrt(1 - rho**2)) - quad

    return ll + log_prior

# === Step (d): MCMC via Metropolis-Hastings ===
def metropolis_hastings(data, num_samples, initial_state, proposal_sd):
    samples = np.zeros((num_samples, 2))
    current = initial_state
    current_log_post = log_joint_posterior(current[0], current[1], data)

    for i in range(num_samples):
        proposal = current + np.random.normal(0, proposal_sd, size=2)
        proposal_log_post = log_joint_posterior(proposal[0], proposal[1], data)

        acceptance_ratio = np.exp(proposal_log_post - current_log_post)
        if np.random.rand() < acceptance_ratio:
            current = proposal
            current_log_post = proposal_log_post

        samples[i] = current

    return samples

# === Step (e): Repeat with different n values ===
def run_experiment(n_values=[10, 100, 1000], num_mcmc_samples=5000):
    for n in n_values:
        print(f"\n--- Running experiment for n = {n} ---")
        theta, psi = sample_from_prior()
        data = generate_data(theta, psi, n)
        init_state = [theta, psi]
        samples = metropolis_hastings(data, num_samples=num_mcmc_samples, initial_state=init_state, proposal_sd=0.1)

        thetas, psis = samples.T

        print(f"True (theta, psi): ({theta:.3f}, {psi:.3f})")
        print(f"Posterior mean: theta = {np.mean(thetas):.3f}, psi = {np.mean(psis):.3f}")

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(thetas, bins=30, density=True)
        plt.title(f"Posterior of Theta (n={n})")

        plt.subplot(1, 2, 2)
        plt.hist(psis, bins=30, density=True)
        plt.title(f"Posterior of Psi (n={n})")
        plt.tight_layout()
        plt.savefig(f"posterior_n_{n}.png")
        plt.show()

if __name__ == "__main__":
    # data = generate_data(1, 1, 10)  # Example to generate data
    # print(data)
    run_experiment([10, 100, 1000])