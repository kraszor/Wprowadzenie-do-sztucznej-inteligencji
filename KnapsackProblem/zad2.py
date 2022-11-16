# Igor Kraszewski
class KnapsacProblemHeuristic:
    def __init__(self, weights, max_weight, values):
        self.weights = weights
        self.max_weight = max_weight
        self.values = values
        self.coeff = self.values / self.weights
        self.result = None
        self.result_weight = None
        self.result_value = None

    def sort_by_coeff(self):
        sorted_list = sorted(range(len(self.coeff)), key=lambda i: -self.coeff[i])
        return sorted_list

    def heuristic_approach(self, sorted_coeffs):
        result = []
        current_weight = 0
        max_value = 0
        for index in sorted_coeffs:
            weight = self.weights[index]
            value = self.values[index]
            if current_weight + weight < self.max_weight:
                result.append((value, weight))
                current_weight += weight
                max_value += value
            else:
                self.result_value = max_value
                self.result_weight = current_weight
                self.result = result
                return result

    def __str__(self):
        outcome = ""
        for pair in self.result:
            outcome = outcome + str(tuple(pair)) + " "
        full_outcome = "Solution are these (value, weight) pairs: " + \
            outcome + f"with value: {self.result_value}" + \
            f" and weight: {self.result_weight}"
        return full_outcome
