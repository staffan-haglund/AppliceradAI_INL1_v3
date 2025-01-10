
import random
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from params import NUMBER_OF_TRUCKS, MAX_POPULATION, MAX_GENERATIONS, MUTATION_RATE, MAX_CAPACITY
from utils import new_screen

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_packages(file):
    '''Generator som läser paket från en CSV-fil'''
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


def initialize_population() -> list:
    ''' Initialiserar population i form av lista, där 1-10 anger vilken lastbil 
        som paketet är i, och 0 är om paketet inte är i någon lastbil'''
    population = []
    for _ in range(MAX_POPULATION):
        chromosome = [0] * num_packages     # Alla paket är ej tilldelade från början
        truck_loads = [[] for _ in range(NUMBER_OF_TRUCKS)]
        truck_weights = [0] * NUMBER_OF_TRUCKS
        
        for i, package in enumerate(packages):
            assigned = False
            possible_trucks = list(range(NUMBER_OF_TRUCKS))
            random.shuffle(possible_trucks)  # Slumpa lastbilarnas ordning
            for truck in possible_trucks:
                if truck_weights[truck] + package['Vikt'] <= MAX_CAPACITY[truck]:
                    chromosome[i] = truck + 1  # Lasta paket i lastbil
                    truck_loads[truck].append(i)
                    truck_weights[truck] += package['Vikt']
                    assigned = True
                    break
            if not assigned:
                chromosome[i] = 0  # Lämna paketet i lagret
        population.append(chromosome)
    return population


def fitness(chromosome) -> int:
    ''' Fitness function
        penalty när lastbil går över kapacitet eller paket är försenat
        högre värde när paket har högre förtjänstvärde
    '''
    truck_loads = [[] for _ in range(NUMBER_OF_TRUCKS)]
    for i, truck in enumerate(chromosome):
        if truck > 0:  # Ignorera olastade paket (0)
            truck_loads[truck - 1].append(i)
    
    total_value = 0
    penalty = 2000  # Penalty om man går över kapacitet
    for truck in truck_loads:
        total_weight = sum(package_info[i]['Vikt'] for i in truck)
        # Om lastbilen ej överlastad, ge högre poäng efter förtjänst
        if total_weight <= MAX_CAPACITY[0]:
            total_value += sum(package_info[i]['Förtjänst'] for i in truck)
        # Om överlastad, dra ifrån tillräckligt stor penalty
        else:
            total_value -= penalty

    # Deadline
    deadline_penalty = 0
    for i, truck in enumerate(chromosome):
        if truck > 0:
            days_until_deadline = package_info[i]['Deadline']
            # Om paketet är försenat, använd straffavgift enligt specifikationer
            if days_until_deadline < 0:
                deadline_penalty = deadline_penalty + -(abs(days_until_deadline) ** 2)

    total_fitness = total_value + deadline_penalty
    return total_fitness


def selection(population):
    ''' Selection som använder Truncation Selection. 
        Funktionen väljer ut den halva av populationen med högst fitness.'''
    population.sort(key=fitness, reverse=True)
    return population[:MAX_POPULATION // 2]


def crossover_1point(parent1, parent2) -> tuple:
    ''' Crossover, one point '''
    point = random.randint(1, num_packages - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def crossover_2point(parent1, parent2) -> tuple:
    ''' Crossover, two point '''
    point1, point2 = sorted(random.sample(range(1, num_packages + 1), 2))
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point1:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point1:]
    return child1, child2


def mutate(chromosome):
    ''' Mutationsfunktion
        Byter ut ett slumpmässigt paket'''
    if random.random() < MUTATION_RATE:
        index = random.randint(0, num_packages - 1)
        chromosome[index] = random.randint(0, NUMBER_OF_TRUCKS)


def calculate_statistics(best_solution):
    ''' Beräkna statistik 
        - Den uppmätta förtjänsten för dagens leveranser.
        - Den totala straffavgiften för paket som är kvar i lager.
        - Antal paket kvar i lager.
        - Den totala förtjänsten kvar i lager (exklusive straffavgiften).
    '''

    # Antal paket kvar i lager
    nr_of_unassigned_packages = best_solution.count(0)

    # Den totala straffavgiften för paket som är kvar i lager.
    # −(x^2)där x är antal dagar efter deadline.
    # unassigned_total_profit = sum(t[2] for t in unassigned_packages)
    unassigned_packages = []
    for i, truck in enumerate(best_solution):
        if truck == 0:
            package = package_info[i]
            unassigned_packages.append((package['Paket_id'], package['Vikt'], package['Förtjänst'], package['Deadline'], i))
    unassigned_total_penalty = sum(-(abs(t[3]) ** 2) for t in unassigned_packages if t[3] < 0)
    sorted_unassigned_packages = unassigned_packages
    sorted_unassigned_packages.sort(key=lambda x: x[4])


    # Totala förtjänsten kvar i lager (exklusive straffavgiften).
    unassigned_total_profit = sum(t[2] for t in unassigned_packages)
    unassigned_total_profit = unassigned_total_profit - unassigned_total_penalty

    # Bästa solution sorterad efter lastbilarna
    sorted_packages_trucks = []
    for i, truck in enumerate(best_solution):
        if truck > 0:
            package = package_info[i]
            sorted_packages_trucks.append((package['Paket_id'], package['Vikt'], package['Förtjänst'], package['Deadline'], truck, i))
    # Sortera listan efter lastbilarna
    sorted_packages_trucks.sort(key=lambda x: x[4])

    # Förtjänsten för dagens leveranser
    todays_total_profit = sum(t[2] for t in sorted_packages_trucks)

    statistics = {'NrOfUnassignedPackages': nr_of_unassigned_packages, 
                  'UnassignedPenalty': unassigned_total_penalty, 
                  'UnassignedProfit': unassigned_total_profit, 
                  'TodaysTotalProfit': todays_total_profit,
                  'SortedAssignedPackages': sorted_packages_trucks,
                  'SortedUnassignedPackages': sorted_unassigned_packages}
    return statistics


def genetic_algorithm():
    ''' Genetiska algoritmens huvudfunktion'''
    population = initialize_population()
    generation_counter = 0

    for _ in range(MAX_GENERATIONS):
        generation_counter = generation_counter + 1
        print(f'Generation #{generation_counter}', end='\r')

        selected = selection(population)
        new_population = []
        while len(new_population) < MAX_POPULATION:
            parents = random.sample(selected, 2)
            child1, child2 = crossover_1point(*parents)
            mutate(child1)
            mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        population = new_population
        # Bästa lösningen
        best_solution = max(population, key=fitness)
    
    return best_solution


def plot_stats(weight_list, xlabel_weight, ylabel_weight, weight_title, profit_list, xlabel_profit, ylabel_profit, profit_title) -> None:
    ''' Funktion för att plotta statistik 
        Statistik på fördelningen av vikt och förtjänst för paketen, 
        både i bilarna och de som är kvar i lager:
        - Histogram.
        - Medelvärde.
        - Varians.
        - Standardavvikelse.
    '''
    weight_mean = np.mean(weight_list)      # Medel
    weight_variance = np.var(weight_list)   # Varians
    weight_std_dev = np.std(weight_list)    # Standardavvikelse
    
    profit_mean = np.mean(profit_list)
    profit_variance = np.var(profit_list)
    profit_std_dev = np.std(profit_list)

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.hist(weight_list, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel(xlabel_weight)
    ax1.set_ylabel(ylabel_weight)
    ax1.set_title(weight_title)

    ax2.hist(profit_list, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel(xlabel_profit)
    ax2.set_ylabel(ylabel_profit)
    ax2.set_title(profit_title)

    # Plotta medel, varians, standardavvikelse
    ax1.axvline(weight_mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Medel: {weight_mean:.2f}')
    ax1.axvline(weight_mean + weight_std_dev, color='green', linestyle='dashed', linewidth=1.5, label=f'Standardavvikelse: {weight_std_dev:.2f}')
    ax1.axvline(weight_mean - weight_std_dev, color='green', linestyle='dashed', linewidth=1.5)
    ax1.legend(title=f'Varians: {weight_variance:.2f}')

    # Plotta medel, varians, standardavvikelse
    ax2.axvline(profit_mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Medel: {profit_mean:.2f}')
    ax2.axvline(profit_mean + profit_std_dev, color='green', linestyle='dashed', linewidth=1.5, label=f'Standardavvikelse: {profit_std_dev:.2f}')
    ax2.axvline(profit_mean - profit_std_dev, color='green', linestyle='dashed', linewidth=1.5)
    ax2.legend(title=f'Varians: {profit_variance:.2f}')

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    file_loaded = False
    file_name = ""
    best_solution = 0
    num_packages = 0
    menu_sel = ""
    while menu_sel != "0":
        print("--------------------------------------------------")
        print("  Lindas Lustfyllda Rederi")
        print("--------------------------------------------------")
        print(f"Inläst fil: {file_name} (Antal paket: {num_packages})")
        print("--------------------------------------------------")
        print("1. Läs in lagerstatus.csv")
        print("2. Läs in lagerstatus2.csv")
        print("3. Läs in lagerstatus3.csv")
        print("4. Läs in lagerstatus4.csv")
        print("--------------------------------------------------")
        print("5. Kör GA")
        print("--------------------------------------------------")
        print("6. Visa statistik")
        print("7. Plot: fördelningen av vikt och förtjänst\n\tför paketen i bilarna")
        print("8. Plot: fördelningen av vikt och förtjänst\n\tför paketen i lager")
        print("--------------------------------------------------")
        print("0. Avsluta")
        print("--------------------------------------------------")
        menu_sel = input("Menyval> ")

        match menu_sel:
            case '1':
                    packages = list(read_packages(file='lagerstatus.csv'))
                    # Lookup dictionary med paketinfo
                    package_info = {i: package for i, package in enumerate(packages)}
                    # print(f'package_info: {package_info}')
                    num_packages = len(packages)
                    file_loaded = True
                    file_name = "lagerstatus.csv"
                    new_screen()
            case '2':
                    packages = list(read_packages(file='lagerstatus2.csv'))
                    # Lookup dictionary med paketinfo
                    package_info = {i: package for i, package in enumerate(packages)}
                    num_packages = len(packages)
                    file_loaded = True
                    file_name = "lagerstatus2.csv"
                    new_screen()
            case '3':
                    packages = list(read_packages(file='lagerstatus3.csv'))
                    # Lookup dictionary med paketinfo
                    package_info = {i: package for i, package in enumerate(packages)}
                    num_packages = len(packages)
                    file_loaded = True
                    file_name = "lagerstatus3.csv"
                    new_screen()
            case '4':
                    packages = list(read_packages(file='lagerstatus4.csv'))
                    # Lookup dictionary med paketinfo
                    package_info = {i: package for i, package in enumerate(packages)}
                    num_packages = len(packages)
                    file_loaded = True
                    file_name = "lagerstatus4.csv"
                    new_screen()
            case '5':
                    if file_loaded:
                        best_solution = genetic_algorithm()
                        statistics = calculate_statistics(best_solution)
                        # print("Best solution:", best_solution)
                        # print("Fitness:", fitness(best_solution))
                    else:
                        new_screen()
                        print("Läs in en fil first")
            case '6':
                    if best_solution != 0:
                        # print(f'best_solution: {best_solution}')
                        # input()
                        print(f'Statistik:')
                        print(f'Förtjänst för dagens leveranser: {statistics['TodaysTotalProfit']}')
                        print(f'Straffavgift för paket i lagret: {statistics['UnassignedPenalty']}')
                        print(f'Antal paket i lagret: {statistics['NrOfUnassignedPackages']}')
                        print(f'Den totala förtjänsten kvar i lager (exkl. straffavgiften): {statistics['UnassignedProfit']}')
                        print("##################################################################")
                        sorted_packages_trucks = statistics['SortedAssignedPackages']
                        for i in range(11):
                            truck_weight = sum(t[1] for t in sorted_packages_trucks if t[4] == i)
                            # truck_no_packages = sum(j for t in sorted_packages_trucks if t[4] == i)
                            print(f'Lastbil #{i}: {truck_weight}')
                        input("[Enter] för menyn")
                        new_screen()
            case '7':
                    if best_solution != 0:
                        print(f'Plot: fördelningen av vikt och förtjänst för paketen i bilarna')
                        input("Any key for plot")
                        sorted_packages_trucks = statistics['SortedAssignedPackages']
                        truck_weight_list = [t[1] for t in sorted_packages_trucks]
                        truck_profit_list = [t[2] for t in sorted_packages_trucks]
                        # print(f'sorted_packages_trucks {sorted_packages_trucks}')
                        # input()
                        plot_stats(weight_list=truck_weight_list, xlabel_weight="Vikt", ylabel_weight="Antal paket", weight_title="Fördelning av vikt för paketen i lastbilarna", 
                                    profit_list=truck_profit_list, xlabel_profit="Förtjänst", ylabel_profit="Antal paket", profit_title="Fördelning av förtjänst för paketen i lastbilarna")
                        # input("[Enter] för menyn")
            case '8':
                    if best_solution != 0:
                        print(f'Plot: fördelningen av vikt och förtjänst för paketen i lager')
                        input("Any key for plot")
                        sorted_unassigned_packages = statistics['SortedUnassignedPackages']
                        unassigned_weight_list = [t[1] for t in sorted_unassigned_packages]
                        unassigned_profit_list = [t[2] for t in sorted_unassigned_packages]
                        # print(f'sorted_unassigned_packages {sorted_unassigned_packages}')
                        # input()
                        plot_stats(weight_list=unassigned_weight_list, xlabel_weight="Vikt", ylabel_weight="Antal paket", weight_title="Fördelning av vikt för paketen i lager", 
                                    profit_list=unassigned_profit_list, xlabel_profit="Förtjänst", ylabel_profit="Antal paket", profit_title="Fördelning av förtjänst för paketen i lager")
                        # input("[Enter] för menyn")
            case _:
                new_screen()

    # 'NrOfUnassignedPackages': nr_of_unassigned_packages, 
    # 'UnassignedPenalty': unassigned_total_penalty, 
    # 'UnassignedProfit': unassigned_total_profit, 
    # 'TodaysTotalProfit': todays_total_profit,
    # 'SortedAssignedPackages': sorted_packages_trucks,
    # 'SortedUnassignedPackages': sorted_unassigned_packages
