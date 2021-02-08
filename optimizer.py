import numpy as np


class Optimizer_stochastic_gradient_descent:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def decay_ajustment(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_parameters(self, layer):
        # with momentum
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * \
                               layer.gradient_weights

            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * \
                               layer.gradient_biases

            layer.biases_momentums = bias_updates

        # without momentum
        else:
            weight_updates = -self.current_learning_rate * layer.gradient_weights
            bias_updates = -self.current_learning_rate * layer.gradient_biases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def increment_iteration(self):
        self.iterations += 1


class Optimizer_adagrad:
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def decay_ajustment(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_parameters(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.gradient_weights ** 2
        layer.bias_cache += layer.gradient_biases ** 2

        layer.weights += - self.current_learning_rate * layer.gradient_weights / (np.sqrt(layer.weight_cache) +
                                                                                  self.epsilon)
        layer.biases += - self.current_learning_rate * layer.gradient_biases / (np.sqrt(layer.bias_cache) +
                                                                                self.epsilon)

    def increment_iteration(self):
        self.iterations += 1
