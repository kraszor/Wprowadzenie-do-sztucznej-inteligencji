from cec2017.functions import f4, f5
from evolutionary import Evolutionary
import numpy as np
import csv


BUDGET = 10000
UPPER_BOUND = 100
POPULATION_SIZE = 20
ELITE = 1
MUTATION_STRENGTH = 1.5
DIMENSIONALITY = 10
iterations = int(BUDGET / POPULATION_SIZE)
population = np.random.uniform(-UPPER_BOUND, UPPER_BOUND,
                               size=(POPULATION_SIZE, DIMENSIONALITY))


def main():
    function = f5
    eval = []
    if function == f4:
        result_file = "results_f4.csv"
    elif function == f5:
        result_file = "result_f5.csv"
    header = ['iterations', 'population_size',
              'mutation_strength', 'elite', 'mean', 'std',
              'best', 'worst']
    for _ in range(25):
        evol = Evolutionary(iterations, population,
                            MUTATION_STRENGTH, ELITE,
                            function, DIMENSIONALITY)
        result = evol.solver()
        eval.append(result[1])
    mean = round(np.mean(eval), 2)
    std = round(np.std(eval), 2)
    best = round(min(eval), 2)
    worst = round(max(eval), 2)
    with open(result_file, 'a') as file:
        dw = csv.DictWriter(file, delimiter='&',
                            fieldnames=header)
        dw.writerow({'iterations': iterations,
                     'population_size': POPULATION_SIZE,
                     'mutation_strength': MUTATION_STRENGTH,
                     'elite': ELITE,
                     'mean': mean,
                     'std': std,
                     'best': best,
                     'worst': worst})


if __name__ == '__main__':
    main()
