# Igor Kraszewski
from gradient_descent import GradientDescent
from plot import Plot
from booth import booth_function
import matplotlib.pyplot as plt
import cec2017
import numpy as np
import textwrap
from cec2017.functions import f1, f2, f3


UPPER_BOUND = 100
DIMENSIONALITY = 10

if __name__ == '__main__':
    steps_iter = ((pow(10, -9), 2000), (pow(10, -18), 1000), (2*pow(10, -9), 2000), (0.07, 150))
    for _ in range(3):
        for i, function in enumerate((f1, f2, f3, booth_function)):
            DIMENSIONALITY = 10
            if function == booth_function:
                DIMENSIONALITY = 2
            X = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=DIMENSIONALITY)
            alpha = steps_iter[i][0]
            solver = GradientDescent(function, 2, steps_iter[i][1])
            result = solver.solve(X, alpha, solver.gradient_fun())
            print(solver)
            plt.figure().set_figheight(6)
            plot_obj = Plot()
            plot_obj.plot_contour(function)
            plot_obj.add_arrows_to_plot(solver.steps)
            np.set_printoptions(precision=2)
            plt.title("\n".join(textwrap.wrap(f'Function {function.__name__},\n Minimum = {round(function(result), 2)},\n Starting point: {X}', 50)))
            # plt.savefig(f'{functison.__name__} - {_}.png')
            plt.show()


