# Igor Kraszewski
from env import w, W, p
from zad1 import KnapsacProblemExau
from zad2 import KnapsacProblemHeuristic


if __name__ == '__main__':
    print("\nExercise 1")
    problem = KnapsacProblemExau(w, W, p)
    problem.exaustive_approach(problem.get_combinations())
    print(problem)
    print("\nExercise 2")
    problem = KnapsacProblemHeuristic(w, W, p)
    problem.heuristic_approach(problem.sort_by_coeff())
    print(problem)
