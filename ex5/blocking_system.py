import numpy as np
import heapq
import scipy.stats as st

class BlockingSystem:
    def __init__(self):
        self.m = 10
        self.n = 100000
        self.mean_service_time = 8.0
        self.mean_time_between_customers = 1.0

    def generate_service_time(self):
        return np.random.exponential(self.mean_service_time)
    
    def generate_interarrival_time(self):
        U = np.random.uniform(0, 1)
        Y = -np.log(U)
        c = 3
        Z = Y + c * (U - 0.5)
        return Z
    
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
    for _ in range(NUM_RUNS):
        system = BlockingSystem()
        num_blocked, num_served = system.simulate()
        fraction_blocked = num_blocked / (num_blocked + num_served)
        blocking_fractions.append(fraction_blocked)

    data = np.array(blocking_fractions)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(NUM_RUNS)

    # 95% confidence interval using t-distribution
    ci_low, ci_high = st.t.interval(0.95, df=NUM_RUNS - 1, loc=mean, scale=std_err)
    print("Blocking System with Control Variete method Simulation Results:")
    print(f"Number of runs: {NUM_RUNS}")
    print(f"Mean blocking fraction: {mean:.5f}")
    print(f"95% confidence interval: ({ci_low:.5f}, {ci_high:.5f})")
