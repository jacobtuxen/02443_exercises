import numpy as np
import heapq
import scipy.stats as st
from tqdm import tqdm

class BlockingSystem:
    def __init__(self, arrival_mode='poisson', predfined_random_numbers=None):
        self.m = 10  # Number of servers
        self.n = 100000  # Total customers
        self.mean_service_time = 8.0
        self.mean_time_between_customers = 1.0
        self.arrival_mode = arrival_mode
        self.predfined_random_numbers = predfined_random_numbers

    def generate_service_time(self):
        return np.random.exponential(self.mean_service_time)
    
    def generate_interarrival_time(self):
        if self.arrival_mode == 'poisson':
            if self.predfined_random_numbers is not None:
                return self.predfined_random_numbers.pop(0)
            return np.random.exponential(self.mean_time_between_customers)
        elif self.arrival_mode == 'hyperexponential':
            # Hyperexponential: mixture of two exponentials
            p1 = 0.8
            lambda1 = 1.5
            lambda2 = 0.5  # Rates adjusted so mean = 1.0
            scale = 0.9333  # Computed as p1/lambda1 + (1-p1)/lambda2
            if self.predfined_random_numbers is not None:
                U = self.predfined_random_numbers.pop(0)
                if U < p1:
                    return (-np.log(self.predfined_random_numbers.pop(0)) / lambda1) / scale
                else:
                    return (-np.log(self.predfined_random_numbers.pop(0)) / lambda2) / scale
            if np.random.random() < p1:
                return np.random.exponential(1/lambda1) / scale
            else:
                return np.random.exponential(1/lambda2) / scale
    
    def simulate(self):
        num_blocked = 0
        num_served = 0
        current_time = 0.0
        busy_servers = 0
        event_queue = []

        next_arrival = current_time + self.generate_interarrival_time()
        heapq.heappush(event_queue, (next_arrival, 'arrival'))

        while num_blocked + num_served < self.n:
            if self.predfined_random_numbers == []:
                breakpoint()
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
    blocking_fractions_diff_cr = []
    blocking_fractions_diff_ncr = []
    for _ in tqdm(range(NUM_RUNS), desc="Running simulations"):
        U = np.random.uniform(0, 1, 300000)
        theta1 = BlockingSystem(arrival_mode='poisson')
        num_blocked_1, num_served_1 = theta1.simulate()
        theta2 = BlockingSystem(arrival_mode='hyperexponential')
        num_blocked_2, num_served_2 = theta2.simulate()

        theta1_ncr = BlockingSystem(arrival_mode='poisson', predfined_random_numbers=U.tolist())
        num_blocked_1_ncr, num_served_1_ncr = theta1_ncr.simulate()
        theta2_ncr = BlockingSystem(arrival_mode='hyperexponential', predfined_random_numbers=U.tolist())
        num_blocked_2_ncr, num_served_2_ncr = theta2_ncr.simulate()
        fraction_blocked_1_ncr = num_blocked_1_ncr / (num_blocked_1_ncr + num_served_1_ncr)
        fraction_blocked_2_ncr = num_blocked_2_ncr / (num_blocked_2_ncr + num_served_2_ncr)
        diff_ncr = abs(fraction_blocked_1_ncr - fraction_blocked_2_ncr)
        blocking_fractions_diff_ncr.append(diff_ncr)

        fraction_blocked_1 = num_blocked_1 / (num_blocked_1 + num_served_1)
        fraction_blocked_2 = num_blocked_2 / (num_blocked_2 + num_served_2)
        diff = abs(fraction_blocked_1 - fraction_blocked_2)
        blocking_fractions_diff_cr.append(diff)
    confidence_interval_cr = st.t.interval(0.95, len(blocking_fractions_diff_cr)-1, loc=np.mean(blocking_fractions_diff_cr), scale=st.sem(blocking_fractions_diff_cr))
    confidence_interval_ncr = st.t.interval(0.95, len(blocking_fractions_diff_ncr)-1, loc=np.mean(blocking_fractions_diff_ncr), scale=st.sem(blocking_fractions_diff_ncr))
    print(f"Mean difference in blocking fractions (CR): {np.mean(blocking_fractions_diff_cr)}")
    print(f"Mean difference in blocking fractions (NCR): {np.mean(blocking_fractions_diff_ncr)}")
    print(f"95% Confidence Interval for CR: {confidence_interval_cr}")
    print(f"95% Confidence Interval for NCR: {confidence_interval_ncr}")