# Igor Kraszewski
from itertools import combinations


class KnapsacProblemExau:

    def __init__(self, weights, max_weight, values):
        self.weights = weights
        self.max_weight = max_weight
        self.values = values
        self.pairs = [pair for pair in zip(self.values, self.weights)]
        self.optimal = None
        self.optimal_value = None
        self.optimal_weight = None

    def get_combinations(self):
        result = []
        for num in range(1, len(self.pairs)+1):
            result += combinations(self.pairs, num)
        return result

    def exaustive_approach(self, combinations):
        max_value = 0
        current_weight = 0
        result = [(0, 0)]
        for elem in combinations:
            weight = 0
            value = 0
            for x in elem:
                weight += x[1]
                value += x[0]
            if value > max_value and weight <= self.max_weight:
                max_value = value
                current_weight = weight
                result = elem
        self.optimal = result
        self.optimal_value = max_value
        self.optimal_weight = current_weight
        return result

    def __str__(self):
        outcome = ""
        for pair in self.optimal:
            outcome = outcome + str(tuple(pair)) + " "
        full_outcome = "Optimal solution are these (value, weight) pairs: " + \
            outcome + f"with value: {self.optimal_value}" + \
            f" and weight: {self.optimal_weight}"
        return full_outcome
