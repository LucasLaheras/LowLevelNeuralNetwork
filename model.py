from activation_function import *
from loss import *
from layers import *
from metrics import *
from optimizer import *


class Sequential:
    def __init__(self, sequence=None, name='my_sequential'):
        self.name = name
        self.sequence = sequence
        self.optimizer =None
        self.loss_activation = None
        if self.sequence is None:
            self.sequence = list()

    def add(self, dense=None, activation=None):
        if dense is not None:
            self.sequence.append(dense)
        if activation is not None:
            self.sequence.append(activation)

    def pop(self):
        self.sequence.pop()

    def compile(self, optimizer=Optimizer_stochastic_gradient_descent(), loss=Activation_Softmax_Loss_CategoricalCrossentropy()):
        self.optimizer = optimizer
        self.loss_activation = loss

    def fit(self, x, y, epochs, step=100):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        for epoch in range(epochs+1):
            for i in range(len(self.sequence)):
                if i == 0:
                    self.sequence[i].forward(x)
                else:
                    self.sequence[i].forward(self.sequence[i-1].output)

            loss = self.loss_activation.forward(self.sequence[len(self.sequence)-1].output, y)
            predictions = np.argmax(self.loss_activation.output, axis=1)

            accuracy = np.mean(predictions==y)

            if not epoch % step:
                print(f'epoch: {epoch}/{epochs} - loss: {loss} - accuracy: {accuracy}')

            self.loss_activation.backward(self.loss_activation.output, y)
            gradient_inputs = self.loss_activation.gradient_inputs
            for i in reversed(range(len(self.sequence))):
                self.sequence[i].backward(gradient_inputs)
                gradient_inputs = self.sequence[i].gradient_inputs
                if isinstance(self.sequence[i], Layer_Dense):
                    self.optimizer.update_parameters(self.sequence[i])