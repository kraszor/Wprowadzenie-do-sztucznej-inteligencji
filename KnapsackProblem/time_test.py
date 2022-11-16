# Igor Kraszewski
from zad1 import KnapsacProblemExau
import numpy as np
import time
from zad2 import KnapsacProblemHeuristic


def test_one_minute_exau(min_items, max_items):
    times = []
    items = []
    for _ in range(25):
        for i in range(min_items, max_items):
            values = np.random.randint(2, 20, size=i)
            weights = np.random.randint(2, 20, size=i)
            max_weight = int(sum(weights)/2)
            problem = KnapsacProblemExau(weights, max_weight, values)
            start = time.time()
            problem.exaustive_approach(problem.get_combinations())
            stop = time.time()
            if stop-start >= 50:
                times.append(stop - start)
                items.append(i)
                break
    return times, items


def test_one_minute_heur(min_items, max_items):
    times = []
    items = []
    for _ in range(25):
        for i in range(min_items, max_items):
            values = np.random.randint(2, 20, size=i)
            weights = np.random.randint(2, 20, size=i)
            max_weight = int(sum(weights)/2)
            problem = KnapsacProblemHeuristic(weights, max_weight, values)
            start = time.time()
            problem.heuristic_approach(problem.sort_by_coeff())
            stop = time.time()
            if stop-start >= 55:
                times.append(stop - start)
                items.append(i)
                break
    return times, items


def test_ratio(min_items, max_items):
    times = np.empty((25, max_items-min_items))
    ratio = np.empty((25, max_items-min_items - 1))
    for _ in range(25):
        for i in range(min_items, max_items):
            values = np.random.randint(2, 20, size=i)
            weights = np.random.randint(2, 20, size=i)
            max_weight = int(sum(weights)/2)
            problem = KnapsacProblemExau(weights, max_weight, values)
            start = time.time()
            problem.exaustive_approach(problem.get_combinations())
            stop = time.time()
            times[_][i-2] = (stop-start)
        ratio[_] = times[_][1:]/times[_][:-1]
    return ratio


if __name__ == '__main__':
    # sprawdzanie ile przedmiotów powoduje czas trwania około minuty
    times, items = test_one_minute_exau(25, 30)
    print("{0:02f}s".format(np.mean(times)), np.mean(items))  # 52 sekundy dla 25 przedmiotów
    times, items = test_one_minute_heur(33500000, 33500001)
    print(times, items)
    print("{0:02f}s".format(np.mean(times)), np.mean(items))
    # sprawdzenie zmiany gdy doda się jeden dodatkowy przedmiot
    ratio = test_ratio(2, 12)
    print("{0:02f}".format(np.mean(ratio)))  # 2,1 razy średio wzrasta czas po dodaniu jednego przedmiotu
