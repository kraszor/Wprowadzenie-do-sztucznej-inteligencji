# Igor Kraszewski
import numpy as np
import matplotlib.pyplot as plt


class Plot:
    def __init__(self):
        self.MAX_X = 100
        self.PLOT_STEP = 0.1

    def plot_contour(self, function):
        x_arr = np.arange(-self.MAX_X, self.MAX_X, self.PLOT_STEP)
        y_arr = np.arange(-self.MAX_X, self.MAX_X, self.PLOT_STEP)
        X, Y = np.meshgrid(x_arr, y_arr)
        Z = np.empty(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = function(np.array([X[i, j], Y[i, j]]))

        plt.contour(X, Y, Z, 20)

    def add_arrows_to_plot(self, gradient_x_steps):
        steps_length = gradient_x_steps[1:] - gradient_x_steps[:-1]
        for index in range(len(steps_length) - 1):
            plt.arrow(gradient_x_steps[index, 0],
                      gradient_x_steps[index, 1],
                      steps_length[index, 0],
                      steps_length[index, 1],
                      head_width=2,
                      head_length=4,
                      fc='k',
                      ec='r')