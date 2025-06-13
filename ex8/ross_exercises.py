import numpy as np

def exercise_13(arr, a, b, sample_size):
    n = len(arr)
    bootstrap_means = np.array([np.mean(np.random.choice(arr, n, replace=True)) for _ in range(sample_size)])
    p_estimate = np.mean((a < bootstrap_means - np.mean(arr)) & (bootstrap_means - np.mean(arr) < b))
    return p_estimate

def exercise_15(arr, sample_size):
    n = len(arr)
    bootstraps_var = np.array([np.var(np.random.choice(arr, n, replace=True), ddof=1) for _ in range(sample_size)])
    S = np.var(bootstraps_var, ddof=1)
    return S

if __name__ == "__main__":
    sample_size = 1000
    arr_13 = np.array([56, 101, 78, 87, 93, 87, 64, 72, 80, 99])
    a, b = -5, 5
    p_estimate = exercise_13(arr_13, a, b, sample_size)
    print(f"Bootstrap probability estimate for exercise 13: {p_estimate:.4f}")

    arr_15 = np.array([5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 16, 8])
    S = exercise_15(arr_15, sample_size)
    print(f"Bootstrap variance estimate for exercise 15: {S:.4f}")