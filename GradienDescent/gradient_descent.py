# Igor Kraszewski
import numpy as np
from autograd import grad


class GradientDescent:
    def __init__(self, fun, dim, max_iter=1000):
        self.fun = fun
        self.dim = dim
        self.max_iter = max_iter
        self.steps = np.empty((self.max_iter, self.dim))
        self.result = None

    def gradient_fun(self):
        gradient = grad(self.fun)
        return gradient

    def solve(self, x, alpha, gradient):
        stan = []
        for i in range(self.max_iter):
            for j in range(self.dim):
                self.steps[i][j] = x[j]
            stan.append(gradient(x))
            x = x - alpha * gradient(x)
        for j in range(self.dim):
            self.steps[-1][j] = x[j]
        self.result = x
        return x

    def __str__(self):
        output = f"Minimum of the function is in point \
{tuple(np.round(self.result, 2))} with value {round(self.fun(self.result),2)}"
        return output
