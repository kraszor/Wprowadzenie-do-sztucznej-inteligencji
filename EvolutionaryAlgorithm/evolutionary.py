# import copy
import numpy as np
# import random


class Evolutionary:
    def __init__(self, iter, population, mutation_strength,
                 elite, q, dim) -> None:
        self.iter = iter
        self.population = population
        self.mutation_strength = mutation_strength
        self.elite = elite
        self.q = q
        self.dim = dim

    def reproduction(self, table):
        result = []

        def rand_pairs():
            ind = int(np.random.choice(len(table), 1))
            return ind

        for _ in range(len(table)):
            first_elem = table[rand_pairs()]
            second_elem = table[rand_pairs()]
            if first_elem[1] <= second_elem[1]:
                result.append(first_elem)
            else:
                result.append(second_elem)
        return result

    def mutation(self, args):
        mutate = self.mutation_strength * np.random.normal(0, 1,
                                                           (len(args),
                                                            self.dim))
        args = np.stack(args) + mutate
        return args

    def succession(self):
        pass

    def solver(self):
        eval = np.apply_along_axis(self.q, 1, self.population)
        table = list(zip(self.population, eval))
        x_hat, y_hat = min(table, key=lambda item: item[1])
        for i in range(self.iter):
            R = self.reproduction(table)
            M = self.mutation(np.array(R, dtype=object)[:, 0])
            eval_inside = np.apply_along_axis(self.q, 1, M)
            table_inside = list(zip(M, eval_inside))
            x_hat_inside, y_hat_inside = min(table_inside,
                                             key=lambda item: item[1])
            if y_hat_inside <= y_hat:
                y_hat = y_hat_inside
                x_hat = x_hat_inside
            table.sort(reverse=False,
                       key=lambda elem: elem[1])
            elite_elem = (np.array(table,
                                   dtype=object)[:self.elite, :]).tolist()
            for elem in elite_elem:
                table_inside.append(elem)
            table_inside.sort(reverse=False,
                              key=lambda elem: elem[1])
            table = (np.array(table_inside,
                              dtype=object)[:-self.elite, :]).tolist()
        return x_hat, y_hat
