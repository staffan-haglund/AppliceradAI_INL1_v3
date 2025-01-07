
import random
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parametrar
# CROSSOVER_RATE = 0.5
NUMBER_OF_TRUCKS = 10
MAX_POPULATION = 200                # Antal kromosomer (möjliga lösningar)
MAX_GENERATIONS = 400
MUTATION_RATE = 0.3
MAX_CAPACITY = [800.0] * NUMBER_OF_TRUCKS       # Maxkapacitet per lastbil


def read_packages(file='lagerstatus1.csv'):
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


# Läs in paket
packages = list(read_packages('lagerstatus1.csv'))
num_packages = len(packages)

# Lookup dictionary med paketinfo
package_info = {i: package for i, package in enumerate(packages)}

# Antagande: Förtjänstkategori 1 är bäst, 10 är sämst
fortjanst = [0, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

# Initialize population
# def initialize_population():
#     return [[random.randint(0, NUMBER_OF_TRUCKS) for _ in range(num_packages)] for _ in range(MAX_POPULATION)]

# Initialisera population utan att överlasta lastbilarna
def initialize_population() -> list:
    ''' Initialiserar population i form av lista, där 1-10 anger vilken lastbil 
        som paketet är i, och 0 är om paketet inte är i någon lastbil'''
    population = []
    for _ in range(MAX_POPULATION):
        chromosome = [0] * num_packages     # Alla paket är unassigned från början
        truck_loads = [[] for _ in range(NUMBER_OF_TRUCKS)]
        truck_weights = [0] * NUMBER_OF_TRUCKS
        
        for i, package in enumerate(packages):
            assigned = False
            possible_trucks = list(range(NUMBER_OF_TRUCKS))
            random.shuffle(possible_trucks)  # Slumpa lastbilarnas ordning
            for truck in possible_trucks:
                if truck_weights[truck] + package['Vikt'] <= MAX_CAPACITY[truck]:
                    chromosome[i] = truck + 1  # Knyt paket till lastbil (1-indexed)
                    truck_loads[truck].append(i)
                    truck_weights[truck] += package['Vikt']
                    assigned = True
                    break
            if not assigned:
                chromosome[i] = 0  # Lämna paketet olastat
        population.append(chromosome)
    return population


# Fitness function med penalty för att gå över kapacitet
def fitness(chromosome) -> int:
    truck_loads = [[] for _ in range(NUMBER_OF_TRUCKS)]
    for i, truck in enumerate(chromosome):
        if truck > 0:  # Ignorera olastade paket (0)
            truck_loads[truck - 1].append(i)
    
    total_value = 0
    penalty = 2000  # Penalty om man går över kapacitet
    for truck in truck_loads:
        total_weight = sum(package_info[i]['Vikt'] for i in truck)
        if total_weight <= MAX_CAPACITY[0]:
            total_value += sum(package_info[i]['Förtjänst'] for i in truck)
        else:
            total_value -= penalty  # Penalty för överlastade lastbilar

    # Deadline
    deadline_penalty = 0
    for i, truck in enumerate(chromosome):
        if truck > 0:  # Ignorera olastade paket (0)
            days_until_deadline = package_info[i]['Deadline']
            if days_until_deadline < 0:
                deadline_penalty = deadline_penalty + -(abs(days_until_deadline) ** 2)

    total_fitness = total_value + deadline_penalty
    return total_fitness

# Selection
def selection(population):
    population.sort(key=fitness, reverse=True)
    return population[:MAX_POPULATION // 2]

# Crossover one point
def crossover_1point(parent1, parent2) -> tuple:
    ''' Crossover, typ one point '''
    point = random.randint(1, num_packages - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Crossover two point
def crossover_2point(parent1, parent2) -> tuple:
    ''' Crossover, typ two point '''
    point1, point2 = sorted(random.sample(range(1, num_packages + 1), 2))
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point1:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point1:]
    return child1, child2

# Mutation
def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, num_packages - 1)
        chromosome[index] = random.randint(0, NUMBER_OF_TRUCKS)

# Den uppmätta förtjänsten för dagens leveranser.
# Den totala straffavgiften för paket som är kvar i lager.
# Antal paket kvar i lager.
# Den totala förtjänsten kvar i lager (exklusive straffavgiften).

def calculate_statistics(stats):
    match stats:
        case 1:     # Förtjänsten för dagens leveranser
            return "zero"
        case 2:     # Straffavgift för paket kvar i lager
            return "one"
        case 3:     # Antal paket kvar i lager
            return best_solution.count(0)
        case 4:     # Total förtjänst kvar i lager
            return "three"
        case default:
            return "something"


# Genetic Algorithm
population = initialize_population()

generation_counter = 0

for _ in range(MAX_GENERATIONS):
    generation_counter = generation_counter + 1
    print(f'Generation #{generation_counter}')

    selected = selection(population)
    new_population = []
    while len(new_population) < MAX_POPULATION:
        parents = random.sample(selected, 2)
        # n = random.randint(0, 1)
        # if n == 0:
        child1, child2 = crossover_1point(*parents)
        # else:
        #     child1, child2 = crossover_2point(*parents)
        mutate(child1)
        mutate(child2)
        new_population.append(child1)
        new_population.append(child2)
    population = new_population

input()

# Bästa(?) lösning
best_solution = max(population, key=fitness)
print("Best solution:", best_solution)
print("Fitness:", fitness(best_solution))
input()

# package_info = {i: package for i, package in enumerate(packages)}
for i, truck in enumerate(best_solution):
    if truck > 0:
        package = package_info[i]
        print(f"Package ID: {package['Paket_id']}, Weight: {package['Vikt']}, Value: {package['Förtjänst']}, Truck: {truck}, i: {i}")

input()

sorted_packages = []
for i, truck in enumerate(best_solution):
    if truck > 0:
        package = package_info[i]
        sorted_packages.append((package['Paket_id'], package['Vikt'], package['Förtjänst'], package['Deadline'], truck, i))
# Sortera listan efter "truck"-värde
sorted_packages.sort(key=lambda x: x[4])

for package_id, weight, value, deadline, truck, i in sorted_packages:
    print(f'Paket_id: {package_id}, Vikt: {weight}, Förtjänst: {value}, Truck: {truck}, i: {i}')

print(f'Antal paket i lastbilar: {len(sorted_packages)}')

print("Fitness:", fitness(best_solution))

print(f'{best_solution.count(0)}')

input()
# truck1_weight = sum(t[1] for t in sorted_packages if t[4] == 10)
total_profit = sum(t[2] for t in sorted_packages)

for i in range(11):
    truck_weight = sum(t[1] for t in sorted_packages if t[4] == i)
    print(f'Lastbil #{i}: {truck_weight}')

print(f'Total förtjänst: {total_profit}')