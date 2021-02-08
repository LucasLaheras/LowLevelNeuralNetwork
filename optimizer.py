import numpy as np


class Optimizer_stochastic_gradient_descent:
    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def decay_ajustment(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_parameters(self, layer):
        layer.weights += -self.current_learning_rate * layer.gradient_weights
        layer.biases += -self.current_learning_rate * layer.gradient_biases

    def increment_iteration(self):
        self.iterations += 1
