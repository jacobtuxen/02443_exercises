import numpy as np
from scipy.stats import t, chi2 #use preimplented t and chi2 distributions
from continous_distribution import normal_distribution_box_mueller

def confidence_intervals_for_mean_variance(samples=100, obs_per_sample=1000, mu=0.0, sigma=1.0):
    alpha = 0.05
    mean_covers = 0
    var_covers = 0

    for _ in range(samples):
        Z0, Z1 = normal_distribution_box_mueller(obs_per_sample // 2, mu, sigma)
        sample = np.concatenate([Z0, Z1])  # total of 10 observations

        sample_mean = np.mean(sample)
        sample_var = np.var(sample, ddof=1)


        t_crit = t.ppf(1 - alpha/2, df=obs_per_sample - 1)
        se = np.sqrt(sample_var / obs_per_sample)
        ci_mean = (sample_mean - t_crit * se, sample_mean + t_crit * se)
        if ci_mean[0] <= mu <= ci_mean[1]:
            mean_covers += 1

        chi2_lower = chi2.ppf(alpha / 2, df=obs_per_sample - 1)
        chi2_upper = chi2.ppf(1 - alpha / 2, df=obs_per_sample - 1)
        ci_var = ((obs_per_sample - 1) * sample_var / chi2_upper,
                  (obs_per_sample - 1) * sample_var / chi2_lower)
        if ci_var[0] <= sigma**2 <= ci_var[1]:
            var_covers += 1

    print(f"Proportion of 95% CI for the mean that contain true mean (μ={mu}): {mean_covers}/{samples}")
    print(f"Proportion of 95% CI for the variance that contain true variance (σ²={sigma**2}): {var_covers}/{samples}")

if __name__ == "__main__":
    confidence_intervals_for_mean_variance()
