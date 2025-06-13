import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def calculate_cost(solution, tsp_matrix):
    cost = 0.0
    n = len(solution)
    for i in range(n):
        cost += tsp_matrix[solution[i], solution[(i + 1) % n]]
    return cost

def calculate_euclidean_cost(solution):
    cost = 0.0
    n = len(solution)
    for i in range(n):
        x1, y1 = solution[i]
        x2, y2 = solution[(i + 1) % n]
        cost += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return cost

def acceptance_probability(current_cost, new_cost, temp):
    if new_cost < current_cost:
        return 1.0
    else:
        return np.exp((current_cost - new_cost) / temp)
    
def temperature_schedule(iteration, initial_temp=1000):
    return 1/np.sqrt(iteration + 1) * initial_temp

def simulated_annealing(tsp_matrix, initial_temp=0.1, max_iterations=10000):
    n = tsp_matrix.shape[0]
    current_solution = np.random.permutation(n-1) + 1
    #make it start and end at 0
    temp = np.insert(current_solution, 0, 0)  # Start at node 0
    temp = np.append(current_solution, 0)  # End at node 0
    current_cost = calculate_cost(temp, tsp_matrix)

    best_solution = np.copy(current_solution)
    best_cost = current_cost

    temp = temperature_schedule(0, initial_temp)

    for iteration in range(max_iterations):
        if temp <= 1e-8:
            break

        new_solution = np.copy(current_solution)
        i, j = np.random.choice(n-1, size=2, replace=False)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        temp_new = np.insert(new_solution, 0, 0)  # Start at node 0
        temp_new = np.append(new_solution, 0)
        new_cost = calculate_cost(temp_new, tsp_matrix)

        if acceptance_probability(current_cost, new_cost, temp) > np.random.rand():
            current_solution = new_solution
            current_cost = new_cost

            if current_cost < best_cost:
                best_solution = np.copy(current_solution)
                best_cost = current_cost

        temp = temperature_schedule(iteration + 1, initial_temp)

    best_solution = np.insert(best_solution, 0, 0)  # Start at node 0
    best_solution = np.append(best_solution, 0)  # End at node 0
    return best_solution, best_cost

def artificial_tsp_matrix(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    tsp_matrix = np.zeros((n, n))
    coordinates = np.column_stack((x, y))
    for i in range(n):
        tsp_matrix[i, 0] = x[i]
        tsp_matrix[i, 1] = y[i]
    for i in range(n):
        for j in range(n):
            tsp_matrix[i, j] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
    return tsp_matrix, coordinates


def plot_solution(tsp_matrix, solution, coordinates=None):
    n = len(solution)
    x = np.zeros(n)
    y = np.zeros(n)

    for i in range(n):
        x[i] = coordinates[solution[i], 0] if coordinates is not None else solution[i]
        y[i] = coordinates[solution[i], 1] if coordinates is not None else tsp_matrix[solution[i], solution[(i + 1) % n]]
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-')
    plt.title('TSP Solution Path')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.plot(x[0], y[0], 'go', label='Start/End Point')
    plt.legend()
    plt.grid()
    plt.show()

def plot_coordinates(coordinates):
    plt.figure(figsize=(10, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', marker='o')
    plt.title('TSP Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
  # tsp_matrix_circle, coordinates_circle = artificial_tsp_matrix(20)
  # best_solution, best_cost = simulated_annealing(tsp_matrix_circle)
  # print("Best solution (circle):", best_solution)
  # print("Best cost (circle):", best_cost)
  # plot_solution(tsp_matrix_circle, best_solution, coordinates_circle)

  tsp_matrix = np.loadtxt("cost.csv", delimiter=",")
  best_solution, best_cost = simulated_annealing(tsp_matrix)
  print("Best solution:", best_solution)
  print("Best cost:", best_cost)
  plot_solution(tsp_matrix, best_solution)
