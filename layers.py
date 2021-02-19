import numpy as np

class Layer_Input:
    def __init__(self, name='input_layer'):
        self.name = name

    def forward(self, inputs, training):
        self.output = inputs


class Layer_Dense:
    def __init__(self, number_inputs, number_neurons, name='my_layer', weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights using random numbers using gaussian distribution and normalized in range [0 0.10]
        self.weights = 0.10 * np.random.randn(number_inputs, number_neurons)
        self.biases = np.zeros((1, number_neurons))

        # Regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

        self.inputs = None
        self.output = None
        self.gradient_weights = None
        self.gradient_biases = None
        self.gradient_inputs = None

        self.name = name

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, gradient):
        self.gradient_weights = np.dot(self.inputs.T, gradient)
        self.gradient_biases = np.sum(gradient, axis=0, keepdims=True)

        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.gradient_weights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.gradient_weights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.gradient_biases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.gradient_biases += 2 * self.bias_regularizer_l2 * self.biases

        self.gradient_inputs = np.dot(gradient, self.weights.T)


class Layer_Dropout:
    def __init__(self, rate, name='my_dropout'):
        self.rate = 1 - rate
        self.inputs = None
        self.gradient_inputs = None
        self.binary_mask = None
        self.output = None
        self.name = name

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, gradient):
        self.gradient_inputs = gradient * self.binary_mask
