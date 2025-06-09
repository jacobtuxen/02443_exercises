import numpy as np
import heapq
import scipy.stats as st  # for confidence intervals
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ex2_3.pareto_composition import exponential_composition
from ex2_3.continous_distribution import pareto_distribution

class BlockingSystem:
    def __init__(self, arrival_distribution = 'poisson', service_distribution = 'exponential'):
        self.m = 10
        self.n = 100000
        self.mean_service_time = 8.0
        self.mean_time_between_customers = 1.0
        self.arrival_distribution = arrival_distribution
        self.service_distribution = service_distribution
        self.k = 2.05  # Shape parameter for Pareto distribution

    def generate_service_time(self):
        if self.service_distribution == 'exponential':
            return np.random.exponential(self.mean_service_time)
        elif self.service_distribution == 'constant':
            return self.mean_service_time
        elif self.service_distribution == 'pareto':
            k = self.k
            beta = (k-1)/k #mean = 1
            return pareto_distribution(samples=1, beta=beta, k=k)[0]
        elif self.service_distribution == 'gamma':
            shape = 2
            scale = self.mean_service_time / shape
            return np.random.gamma(shape, scale)
    
    def generate_interarrival_time(self):
        if self.arrival_distribution == 'poisson':
          return np.random.exponential(self.mean_time_between_customers)
        elif self.arrival_distribution == 'erlang':
          return np.random.gamma(2, self.mean_time_between_customers / 2)
        elif self.arrival_distribution == 'exponential':
          return exponential_composition(samples=1, lambdas=[0.8333, 5.0], probabilities=[0.8, 0.2])[0]
    
    def simulate(self):
        num_blocked = 0
        num_served = 0
        current_time = 0.0
        busy_servers = 0
        event_queue = []

        next_arrival = current_time + self.generate_interarrival_time()
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

                next_arrival = current_time + self.generate_interarrival_time()
                heapq.heappush(event_queue, (next_arrival, 'arrival'))

            elif event_type == 'departure':
                busy_servers -= 1

        return num_blocked, num_served


if __name__ == "__main__":
    NUM_RUNS = 30 
    blocking_fractions = []
    distribution = 'poisson'  # Choose from ['poisson', 'erlang', 'exponential']
    service_distribution = 'gamma'  # Choose from ['exponential', 'constant', 'pareto', 'gamma']
    k = 1.05

    for _ in range(NUM_RUNS):
        system = BlockingSystem(arrival_distribution=distribution)
        num_blocked, num_served = system.simulate()
        fraction_blocked = num_blocked / (num_blocked + num_served)
        blocking_fractions.append(fraction_blocked)

    data = np.array(blocking_fractions)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(NUM_RUNS)

    # 95% confidence interval using t-distribution
    ci_low, ci_high = st.t.interval(0.95, df=NUM_RUNS - 1, loc=mean, scale=std_err)

    print(f"Arrival distribution: {distribution}")
    print(f"Service distribution: {service_distribution}")
    if service_distribution == 'pareto':
        print(f"Using Pareto distribution with k={k}")
    print(f"Number of runs: {NUM_RUNS}")
    print(f"Mean blocking fraction: {mean:.5f}")
    print(f"95% confidence interval: ({ci_low:.5f}, {ci_high:.5f})")
