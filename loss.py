import numpy as np


class Loss:
    def __init__(self):
        self.output = None
        self.data_loss = None
        self.gradient_inputs = None

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss


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
