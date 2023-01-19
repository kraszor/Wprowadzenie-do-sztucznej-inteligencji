#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:51:50 2021

@author: RafaĹ Biedrzycki
Kodu tego mogÄ uĹźywaÄ moi studenci na Äwiczeniach
z przedmiotu WstÄp do Sztucznej Inteligencji.
Kod ten powstaĹ aby przyspieszyÄ i uĹatwiÄ
pracÄ studentĂłw, aby mogli skupiÄ siÄ
na algorytmach sztucznej inteligencji.
Kod nie jest wzorem dobrej jakoĹci programowania
w Pythonie, nie jest rĂłwnieĹź wzorem programowania
obiektowego, moĹźe zawieraÄ bĹÄdy.

Nie ma obowiÄzku uĹźywania tego kodu.
"""

import numpy as np
import matplotlib.pyplot as plt

# ToDo tu prosze podac ostatnie cyfry numerow indeksow
p = [4, 6]

L_BOUND = -5
U_BOUND = 5


def q(x):
    return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))


x = np.linspace(L_BOUND, U_BOUND, 100)
y = q(x)

np.random.seed(1)


# f logistyczna jako przykĹad sigmoidalej
def sigmoid(x):
    return 1/(1+np.exp(-x))


# pochodna fun. 'sigmoid'
def d_sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s * (1-s)


# f. straty
def nloss(y_out, y):
    return (y_out - y) ** 2


# pochodna f. straty
def d_nloss(y_out, y):
    return 2*(y_out - y)

def shuffle(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]


class DlNet:
    def __init__(self, x, y, hidden_l_size=None, lr=None, batch_size=None):
        self.x = x
        self.y = y
        self.y_out = 0
        self.weights = None
        self.batch_size = batch_size if batch_size else 1
        self.HIDDEN_L_SIZE = hidden_l_size if hidden_l_size else 13
        self.LR = lr if lr else 1e-2

    def random_init(self, layers_in, layers_out):
        # generate weights in range (-epsilon, epsilon). Xavier initialization.
        epsilon_init = np.sqrt(6/(layers_in + layers_out))
        w = np.random.rand(layers_out, 1 + layers_in) * 2 \
            * epsilon_init - epsilon_init
        return w

    def init_weights(self):
        w_l = []
        w_l.append(self.random_init(1, self.HIDDEN_L_SIZE))
        w_l.append(self.random_init(self.HIDDEN_L_SIZE, 1))
        self.weights = w_l

    def forward(self, x):
        x = np.array([1, x])
        output = []
        layer = []
        for i in range(2):
            for j in range(self.HIDDEN_L_SIZE):
                if i == 1:
                    output.append([(self.weights[i][j]).dot(x)])
                    return output
                else:
                    layer.append((self.weights[i][j]).dot(x))
            output.append(layer)
            x = [1] + list(sigmoid(np.asarray(layer)))
            x = np.array(x)
            layer = []

    def predict(self, x):
        return self.forward(x)[1][0]

    def backward(self, x, y):
        a1 = np.array([1, x])
        z2 = self.weights[0].dot(a1)
        a2 = sigmoid(z2).tolist()
        a2 = np.asarray([1] + a2)
        a3 = self.weights[1].dot(a2)
        error = d_nloss(a3, y)
        delta = self.weights[1][0, 1:] * error * d_sigmoid(z2)
        return (delta, delta*x, error, error*sigmoid(z2))
    
    def batch(self, x_set, y_set):
        x_set, y_set = shuffle(x_set, y_set)
        min_batch_iter = len(x_set) // self.batch_size
        for i in range(min_batch_iter):
            start = i * self.batch_size
            end = (i+1) *self.batch_size
            self.min_batch(x_set[start:end], y_set[start:end])
    
    def min_batch(self, x_set, y_set):
        vals = [0, 0, 0, 0] # b1, w1, b2, w2
        for x, y in zip(x_set, y_set):
            temp = self.backward(x, y)
            for i in range(len(vals)):
                vals[i] += temp[i]
        for i in vals:
            i *= self.LR / len(x_set)
        b1, w1, b2, w2 = vals
        self.weights[0][:, 0] -= b1
        self.weights[0][:, 1] -= w1
        self.weights[1][0, 0] -= b2
        self.weights[1][0, 1:] -= w2    

    def train(self, x_set, y_set, iters):
        for i in range(0, iters):
            self.batch(x_set, y_set)
            if i % 100 == 0:
                y_hat = []
                for elem in x_set:
                    y_hat.append(self.predict(elem))
                err = nloss(np.asarray(y_hat), y_set)
                err = np.sum(err)
                print(f"Iter: {i} Error: {err}")
        print(err)

def create_plot(x, y, yh, hidden_l_size, lr):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x, y, 'r')
    plt.plot(x, yh, 'b')

    fig.suptitle(f"Hidden layout size: {hidden_l_size}, learning rate: {lr}")
    plt.show()

if __name__ == "__main__":
    nn = DlNet(x, y, hidden_l_size=13, lr=0.05, batch_size=48)
    nn.init_weights()
    nn.train(x, y, 20000)
    yh = []  # ToDo tu umiesciÄ wyniki (y) z sieci
    for elem in x:
        yh.append(nn.predict(elem))
    print("Mean absolute error: ", np.sum(np.abs(y-yh))/len(y))
    create_plot(x, y, yh, hidden_l_size=13, lr=nn.LR)
