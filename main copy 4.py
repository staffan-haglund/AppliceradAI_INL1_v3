
import random
import csv
import os
# import pandas as pd



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NUMBER_OF_GENES = 10                # Antal lastbilar
NUMBER_OF_TRUCKS = 10                # Antal lastbilar
MAX_CAPACITY = 80.0                # Maxkapacitet per lastbil
MAX_POPULATION = 30                 # Antal kromosomer (möjliga lösningar)
MAX_INIT_POPULATION = 20#100           # Antal kromosomer (möjliga lösningar)
MAX_GENERATIONS = 10                # Antal generationer
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.1


def package_generator(file='lagerstatus-2000.csv'):
    '''Generator som läser paket från en CSV-fil och yield ett paket åt gången.'''
    csv_file_path = os.path.join(BASE_DIR, file)
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        # next(reader)                # Hoppa över headern
        for row in reader:
            # Paket_id,Vikt,Förtjänst,Deadline
            package_id = str(row['Paket_id'])
            weight = float(row['Vikt'])
            profit = int(row['Förtjänst'])
            days_until_deadline = int(row['Deadline'])
            yield {'Paket_id': package_id, 'Vikt': weight, 'Förtjänst': profit, 'Deadline': days_until_deadline}


# def initialize_population():
#     return [
#         {item_id: random.randint(0, num_knapsacks) for item_id in item_ids}
#         for _ in range(population_size)
#     ]


def initialize_population():
    population = []
    for _ in range(MAX_INIT_POPULATION):
        # Start with empty knapsacks
        solution = {}
        knapsack_weights = [0] * NUMBER_OF_TRUCKS

        for item in packages:
            item_id = item["Paket_id"]
            item_weight = item["Vikt"]

            # Attempt to assign the item to a random knapsack
            valid_knapsacks = [
                k for k in range(1, NUMBER_OF_TRUCKS + 1)  # Knapsack IDs: 1 to NUMBER_OF_TRUCKS
                if knapsack_weights[k - 1] + item_weight < (MAX_CAPACITY)
            ]

            if valid_knapsacks:
                # Randomly assign to one of the valid knapsacks
                knapsack = random.choice(valid_knapsacks)
                solution[item_id] = knapsack
                knapsack_weights[knapsack - 1] += item_weight
            else:
                # Leave unassigned if no valid knapsack
                solution[item_id] = 0

        population.append(solution)
    return population



# Fitness function
def fitness(solution):
    knapsack_weights = [0] * NUMBER_OF_TRUCKS
    knapsack_values = [0] * NUMBER_OF_TRUCKS
    penalty = 0

    for item_id, knapsack in solution.items():
        if knapsack > 0:  # If assigned
            item = next(item for item in packages if item["Paket_id"] == item_id)
            knapsack_weights[knapsack - 1] += item["Vikt"]
            knapsack_values[knapsack - 1] += item["Förtjänst"]

    for w in knapsack_weights:
        if w > MAX_CAPACITY:
            penalty += (w - MAX_CAPACITY) ** 2

    return sum(knapsack_values) - penalty

# Selection
# def select(population):
#     return random.choices(population, weights=[fitness(p) for p in population], k=2)
def select_parents(population):
    weights = [fitness(p) for p in population]
    if sum(weights) <= 0:
        # Handle invalid or zero weights gracefully
        weights = [1] * len(population)  # Assign equal probability if all weights are zero
    return random.choices(population, weights=weights, k=2)


# Crossover
def crossover(parent1, parent2):
    child = {}
    for item_id in package_ids:
        if random.random() < CROSSOVER_RATE:
            child[item_id] = parent1[item_id]
        else:
            child[item_id] = parent2[item_id]
    return child

def one_point_crossover(parent1, parent2):
    # Choose a random crossover point
    crossover_point = random.randint(1, len(parent1) - 1)
    
    # Create the child by combining genes from both parents
    child = {}
    for i, item_id in enumerate(parent1):
        if i < crossover_point:
            child[item_id] = parent1[item_id]
        else:
            child[item_id] = parent2[item_id]
    return child

# Mutation
def mutate(solution):
    for item_id in package_ids:
        if random.random() < MUTATION_RATE:
            solution[item_id] = random.randint(0, NUMBER_OF_TRUCKS)
    return solution


packages = list(package_generator(file="lagerstatus-2000.csv"))
package_ids = [item['Paket_id'] for item in packages]

print(f'Packages: {packages}')
input()
# print("########################################################")
# print(f'Package_ids: {package_ids}')
population = initialize_population()
input()
print(f'Population: {population}')
input()
# Verify population
for solution in population:
    knapsack_weights = [0] * NUMBER_OF_TRUCKS
    for item_id, knapsack in solution.items():
        if knapsack > 0:
            item = next(item for item in packages if item["Paket_id"] == item_id)
            knapsack_weights[knapsack - 1] += item["Vikt"]
    print("Knapsack weights:", knapsack_weights)
input()

generation_counter = 0
for _ in range(MAX_GENERATIONS):
    generation_counter = generation_counter + 1
    print(f'Generation #{generation_counter}')
    new_population = []
    for _ in range(MAX_INIT_POPULATION // 2):
        print("SELECT")
        # parent1, parent2 = select(population)
        parent1, parent2 = select_parents(population)
        print("CROSSOVER")
        child1, child2 = one_point_crossover(parent1, parent2), one_point_crossover(parent1, parent2)
        new_population.append(mutate(child1))
        new_population.append(mutate(child2))
    population = new_population

# Best solution
best_solution = max(population, key=fitness)
# Verify population
for solution in population:
    knapsack_weights = [0] * NUMBER_OF_TRUCKS
    for item_id, knapsack in solution.items():
        if knapsack > 0:
            item = next(item for item in packages if item["Paket_id"] == item_id)
            knapsack_weights[knapsack - 1] += item["Vikt"]
    print("Knapsack weights:", knapsack_weights)

print("Best solution:", best_solution)
print("Best fitness:", fitness(best_solution))