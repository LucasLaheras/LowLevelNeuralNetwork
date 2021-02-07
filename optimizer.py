import numpy as np


class Optimizer_stochastic_gradient_descent:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_parameters(self, layer):
        layer.weights += -self.learning_rate * layer.gradient_weights
        layer.biases += -self.learning_rate * layer.gradient_biases