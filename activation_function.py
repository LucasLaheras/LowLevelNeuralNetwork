import numpy as np
from loss import *

class Activation_softmax:
    def __init__(self):
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normalized_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = normalized_values

    def backward(self, gradient):
        self.gradient_inputs = np.empty_like(gradient)

        for index, (single_output, single_derivate) in enumerate(zip(self.output, gradient)):

            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_derivate.T)

            self.gradient_inputs[index] = np.dot(jacobian_matrix, single_derivate)


class Activation_ReLU:
    def __init__(self):
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, gradient):
        self.gradient_inputs = gradient.copy()

        self.gradient_inputs[self.inputs < 0] = 0


class Activation_Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):
        self.output = None
        self.gradient_inputs = None
        self.activation = Activation_softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, gradient, y_true):
        samples = len(gradient)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.gradient_inputs = gradient.copy()
        self.gradient_inputs[range(samples), y_true] -= 1
        self.gradient_inputs = self.gradient_inputs / samples
