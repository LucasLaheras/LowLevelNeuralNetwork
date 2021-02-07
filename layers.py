import numpy as np


class Layer_Dense:
    def __init__(self, number_inputs, number_neurons, name='my_layer'):
        self.weights = 0.10 * np.random.randn(number_inputs, number_neurons) # Random numbers using gaussian distribution and \
        # normalized in range [0 1]
        self.biases = np.zeros((1, number_neurons))

        self.inputs = None
        self.output = None
        self.gradient_weights = None
        self.gradient_biases = None
        self.gradient_inputs = None

        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, gradient):
        self.gradient_weights = np.dot(self.inputs.T, gradient)
        self.gradient_biases = np.sum(gradient, axis=0, keepdims=True)

        self.gradient_inputs = np.dot(gradient, self.weights.T)
