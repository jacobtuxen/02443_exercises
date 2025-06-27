import numpy as np
import heapq
import scipy.stats as st

class BlockingSystem:
    def __init__(self, arrival_distribution = 'poisson', service_distribution = 'exponential'):
        self.m = 10
        self.n = 100000
        self.mean_service_time = 8.0
        self.mean_time_between_customers = 1.0
        self.arrival_distribution = arrival_distribution
        self.service_distribution = service_distribution

    def generate_service_time(self):
        if self.service_distribution == 'exponential':
            return np.random.exponential(self.mean_service_time)
    
    def generate_interarrival_time(self):
        if self.arrival_distribution == 'poisson':
          return np.random.exponential(self.mean_time_between_customers)
    
    def simulate(self):
        num_blocked = 0
        num_served = 0
        current_time = 0.0
        busy_servers = 0
        event_queue = []
        interarrival_times = []

        next_interarrival = self.generate_interarrival_time()
        next_arrival = current_time + next_interarrival
        interarrival_times.append(next_interarrival)
        heapq.heappush(event_queue, (next_arrival, 'arrival'))

        while num_blocked + num_served < self.n:
            event_time, event_type = heapq.heappop(event_queue)
            current_time = event_time

            if event_type == 'arrival':
                if busy_servers < self.m:
                    busy_servers += 1
                    service_time = self.generate_service_time()
                    departure_time = current_time + service_time
                    heapq.heappush(event_queue, (departure_time, 'departure'))
                    num_served += 1
                else:
                    num_blocked += 1

                next_interarrival = self.generate_interarrival_time()
                next_arrival = current_time + next_interarrival
                interarrival_times.append(next_interarrival)
                heapq.heappush(event_queue, (next_arrival, 'arrival'))

            elif event_type == 'departure':
                busy_servers -= 1

        return num_blocked, num_served, interarrival_times


if __name__ == "__main__":
    NUM_RUNS = 30
    blocking_fractions = []
    avg_interarrivals = []

    for _ in range(NUM_RUNS):
        system = BlockingSystem(arrival_distribution='poisson', service_distribution='exponential')
        num_blocked, num_served, interarrival_times = system.simulate()

        X = num_blocked / (num_blocked + num_served)
        Y = np.mean(interarrival_times)

        blocking_fractions.append(X)
        avg_interarrivals.append(Y)

    X_vals = np.array(blocking_fractions)
    Y_vals = np.array(avg_interarrivals)
    mu_Y = 1.0

    # Estimate optimal c
    cov_XY = np.cov(X_vals, Y_vals, ddof=1)[0, 1]
    var_Y = np.var(Y_vals, ddof=1)
    c = -(cov_XY / var_Y)

    X_cv = X_vals - c * (Y_vals - mu_Y)

    mean_cv = np.mean(X_cv)
    std_err_cv = np.std(X_cv, ddof=1) / np.sqrt(NUM_RUNS)
    ci_low_cv, ci_high_cv = st.t.interval(0.95, df=NUM_RUNS - 1, loc=mean_cv, scale=std_err_cv)

    print("Blocking System with Control Variates:")
    print(f"Number of runs: {NUM_RUNS}")
    print(f"Mean blocking fraction (control variates): {mean_cv:.5f}")
    print(f"95% confidence interval: ({ci_low_cv:.5f}, {ci_high_cv:.5f})")
    print(c)
