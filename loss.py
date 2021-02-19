import numpy as np


class Loss:
    def __init__(self):
        self.output = None
        self.data_loss = None
        self.gradient_inputs = None
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def regularization_loss(self):
        regularization_loss = 0

        for layer in self.trainable_layers:
            # L1 regularization - weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            # L1 regularization - biases
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, regularization=False):
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, regularization=False):

        data_loss = self.accumulated_sum / self.accumulated_count

        if not regularization:
            return data_loss

        return data_loss, self.regularization_loss()


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip data on both sides to not have log(0) and preventing the average from changing
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        correct_confidences = 1

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        probabilities = - np.log(correct_confidences)
        self.output = probabilities
        return probabilities

    def backward(self, gradient, y_true):
        samples = len(gradient)

        labels = len(gradient[0])

        if len(y_true.shape) == 1:

            list_zeros = np.zeros(labels)
            list_zeros[y_true] = 1
            y_true = list_zeros

        self.gradient_inputs = -y_true / gradient
        self.gradient_inputs = self.gradient_inputs / samples


class Loss_BinaryCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        self.output = sample_losses
        return sample_losses

    def backward(self, gradient, y_true):

        samples = len(gradient)
        outputs = len(gradient[0])

        clipped_gradient = np.clip(gradient, 1e-7, 1 - 1e-7)

        self.gradient_inputs = -(y_true / clipped_gradient - (1 - y_true) / (1 - clipped_gradient)) / outputs
        self.gradient_inputs = self.gradient_inputs / samples


class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        self.output = sample_losses
        return sample_losses

    def backward(self, gradient, y_true):
        samples = len(gradient)

        outputs = len(gradient[0])

        self.gradient_inputs = -2 * (y_true - gradient) / outputs
        self.gradient_inputs = self.gradient_inputs / samples


class LossMeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        self.output = sample_losses
        return sample_losses

    def backward(self, gradient, y_true):
        samples = len(gradient)
        outputs = len(gradient[0])

        self.gradient_inputs = np.sign(y_true - gradient) / outputs
        self.gradient_inputs = self.gradient_inputs / samples