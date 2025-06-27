import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Global constants
A1, A2 = 4, 4
m = 10
c = 1

def joint_prob(i, j):
    if i < 0 or j < 0 or i + j > m:
        return 0
    return (A1**i / math.factorial(i)) * (A2**j / math.factorial(j))

def proposal_2d(current):
    """Random walk in 2D"""
    i, j = current
    new_i = i + np.random.choice([-1, 0, 1])
    new_j = j + np.random.choice([-1, 0, 1])
    return (new_i, new_j)

def metropolis_hastings_joint(num_samples=30000):
    samples = []
    current = (3, 3)
    for _ in range(num_samples):
        proposal = proposal_2d(current)
        pi_current = joint_prob(*current)
        pi_proposal = joint_prob(*proposal)
        alpha = min(1, pi_proposal / (pi_current + 1e-10))
        if np.random.rand() < alpha:
            current = proposal
        samples.append(current)
    return samples

def metropolis_hastings_coordinate(num_samples=30000):
    samples = []
    current = [3, 3]
    for _ in range(num_samples):
        for dim in [0, 1]:
            proposal = current.copy()
            proposal[dim] += np.random.choice([-1, 0, 1])
            if proposal[0] >= 0 and proposal[1] >= 0 and sum(proposal) <= m:
                pi_current = joint_prob(*current)
                pi_proposal = joint_prob(*proposal)
                alpha = min(1, pi_proposal / (pi_current + 1e-10))
                if np.random.rand() < alpha:
                    current = proposal
        samples.append(tuple(current))
    return samples

def gibbs_sampling(num_samples=30000):
    samples = []
    i, j = 0, 0
    for _ in range(num_samples):
        # Sample i | j
        i_vals = [k for k in range(m + 1 - j)]
        probs_i = np.array([A1**k / math.factorial(k) for k in i_vals])
        probs_i /= probs_i.sum()
        i = np.random.choice(i_vals, p=probs_i)

        # Sample j | i
        j_vals = [k for k in range(m + 1 - i)]
        probs_j = np.array([A2**k / math.factorial(k) for k in j_vals])
        probs_j /= probs_j.sum()
        j = np.random.choice(j_vals, p=probs_j)

        samples.append((i, j))
    return samples

def plot_joint_hist(samples, title):
    counts = Counter(samples)
    heatmap = np.zeros((m+1, m+1))
    for (i, j), count in counts.items():
        heatmap[i, j] = count
    plt.imshow(heatmap, origin='lower')
    plt.title(title)
    plt.xlabel('j')
    plt.ylabel('i')
    plt.colorbar(label='Frequency')
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def chi2_test(samples):
    counts = Counter(samples)
    total = sum(counts.values())
    expected = {}
    Z = sum(joint_prob(i, j) for i in range(m+1) for j in range(m+1) if i + j <= m)
    for i in range(m+1):
        for j in range(m+1):
            if i + j <= m:
                expected[(i, j)] = joint_prob(i, j) / Z * total
    obs = []
    exp = []
    for key in expected:
        obs.append(counts.get(key, 0))
        exp.append(expected[key])
    chi2, p = chisquare(obs, f_exp=exp)
    print(f"Chi-squared = {chi2:.2f}, p = {p:.4f}")
    return chi2, p

if __name__ == "__main__":
    np.random.seed(42)

    print("=== Metropolis-Hastings (Joint) ===")
    mh_samples = metropolis_hastings_joint()
    plot_joint_hist(mh_samples, "Metropolis-Hastings (Joint Proposal)")
    chi2_test(mh_samples)

    print("=== Metropolis-Hastings (Coordinate-wise) ===")
    mh_coord_samples = metropolis_hastings_coordinate()
    plot_joint_hist(mh_coord_samples, "Metropolis-Hastings (Coordinate-wise)")
    chi2_test(mh_coord_samples)

    print("=== Gibbs Sampling ===")
    gibbs_samples = gibbs_sampling()
    plot_joint_hist(gibbs_samples, "Gibbs Sampling")
    chi2_test(gibbs_samples)
