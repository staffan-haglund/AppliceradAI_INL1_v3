
import random
import csv
import os
# import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NUMBER_OF_GENES = 10                # Antal lastbilar
# NUMBER_OF_TRUCKS = 10                # Antal lastbilar
# MAX_CAPACITY = 80.0                # Maxkapacitet per lastbil
# MAX_POPULATION = 30                 # Antal kromosomer (möjliga lösningar)
MAX_INIT_POPULATION = 20#100           # Antal kromosomer (möjliga lösningar)
MAX_GENERATIONS = 300                # Antal generationer
CROSSOVER_RATE = 0.5
# MUTATION_RATE = 0.1

import csv
import random

# Define parameters
NUMBER_OF_TRUCKS = 10
MAX_POPULATION = 100
MAX_GENERATIONS = 600
MUTATION_RATE = 0.1
MAX_CAPACITY = [800.0] * NUMBER_OF_TRUCKS

def read_packages(file='lagerstatus-2000.csv'):
    '''Generator som läser paket från en CSV-fil och yield ett paket åt gången.'''
    csv_file_path = os.path.join(BASE_DIR, file)
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        # next(reader)                # Hoppa över headern
        for row in reader:
            # Paket_id,Vikt,Förtjänst,Deadline
            package_id = str(row['Paket_id'])
            weight = float(row['Vikt'])
            profit = int(row['Förtjänst'])
            days_until_deadline = int(row['Deadline'])
            yield {'Paket_id': package_id, 'Vikt': weight, 'Förtjänst': profit, 'Deadline': days_until_deadline}


# Load packages
packages = list(read_packages('lagerstatus-2000.csv'))
num_packages = len(packages)

# Initialize population
# def initialize_population():
#     return [[random.sample(range(num_packages), num_packages) for _ in range(NUMBER_OF_TRUCKS)] for _ in range(MAX_POPULATION)]

def initialize_population():
    print("INITIALIZE POPULATION")
    population = []
    population_counter = 0
    for _ in range(MAX_POPULATION):
        population_counter = population_counter + 1
        print(f'population_counter: {population_counter}')
        chromosome = [[] for _ in range(NUMBER_OF_TRUCKS)]
        package_counter = 0
        for package in packages:
            package_counter = package_counter + 1
            print(f'package_counter: {package_counter}')
            while True:
                truck_index = random.randint(0, NUMBER_OF_TRUCKS - 1)
                print(truck_index)
                if sum(p['Vikt'] for p in chromosome[truck_index]) + package['Vikt'] < MAX_CAPACITY[truck_index]:
                    chromosome[truck_index].append(package)
                    break
                else:
                    break
        population.append(chromosome)
    return population
    

# Fitness function with penalty for exceeding capacity
def fitness(chromosome):
    total_value = 0
    penalty = 1000  # Penalty for exceeding capacity
    for truck in chromosome:
        total_weight = sum(package['Vikt'] for package in truck)
        if total_weight <= MAX_CAPACITY[0]:
            total_value += sum(package['Förtjänst'] for package in truck)
        else:
            total_value -= penalty  # Apply penalty for overweight trucks
    return total_value


# Selection
def selection(population):
    population.sort(key=fitness, reverse=True)
    return population[:MAX_POPULATION // 2]

# Crossover
def crossover(parent1, parent2):
    point = random.randint(1, num_packages - 1)
    child1 = [parent1[i] if i < point else parent2[i] for i in range(NUMBER_OF_TRUCKS)]
    child2 = [parent2[i] if i < point else parent1[i] for i in range(NUMBER_OF_TRUCKS)]
    return child1, child2


# Mutation
def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(NUMBER_OF_TRUCKS), 2)
        k, l = random.randint(0, len(chromosome[i]) - 1), random.randint(0, len(chromosome[j]) - 1)
        chromosome[i][k], chromosome[j][l] = chromosome[j][l], chromosome[i][k]


# Genetic Algorithm
population = initialize_population()
# for pop in population:
#     input()
#     print(f'POPULATION: {pop}')

generation_counter = 0

for _ in range(MAX_GENERATIONS):
    generation_counter = generation_counter + 1
    print(f'Generation #{generation_counter}')

    selected = selection(population)
    new_population = []
    while len(new_population) < MAX_POPULATION:
        parents = random.sample(selected, 2)
        child1, child2 = crossover(*parents)
        mutate(child1)
        mutate(child2)
        new_population.append(child1)
        new_population.append(child2)
    population = new_population

input()
# Best solution
best_solution = max(population, key=fitness)

truck_counter = 0
for truck in best_solution:
    truck_counter = truck_counter + 1
    print(f'Truck #{truck_counter}: {truck}')
    t_weight = 0
    for t in truck:
        t_weight = t_weight + t['Vikt']
    print(f'Vikt: {t_weight}')
    input()

print("Best solution:", best_solution)
print("Fitness:", fitness(best_solution))

