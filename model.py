from activation_function import *
from loss import *
from layers import *
from metrics import *
from optimizer import *
from activation_function import *
import pickle
import copy


class Sequential:
    def __init__(self, sequence=None, name='my_sequential'):
        self.name = name

        self.sequence = sequence
        if self.sequence is None:
            self.sequence = list()

        self.optimizer = None
        self.loss_function = None
        self.metrics = None
        self.trainable_layers = list()
        self.softmax_classifier_output = None

    def add(self, layer):
        self.sequence.append(layer)

    def pop(self):
        self.sequence.pop()

    def get_layer(self, name=None, index=None):
        if name:
            for layer in self.sequence:
                if layer.name == name:
                    return layer
        if index:
            return self.sequence[index]

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss_function = loss
        self.metrics = metrics
        self.finalize()

    def finalize(self):
        # Create and set the input layer
        self.input_layer = Layer_Input()

        for i in range(len(self.sequence)):
            # If it's the first layer, the previous layer object is the input layer
            if i == 0:
                self.sequence[i].prev = self.input_layer
                self.sequence[i].next = self.sequence[i + 1]
            # All layers except for the first and the last
            elif i < len(self.sequence) - 1:
                self.sequence[i].prev = self.sequence[i - 1]
                self.sequence[i].next = self.sequence[i + 1]
            # The last layer - the next object is the loss
            else:
                self.sequence[i].prev = self.sequence[i - 1]
                self.sequence[i].next = self.loss_function
                self.output_layer_activation = self.sequence[i]

            if hasattr(self.sequence[i], 'weights'):
                self.trainable_layers.append(self.sequence[i])

        self.loss_function.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.sequence[- 1], Activation_softmax) and \
            isinstance(self.loss_function, Loss_CategoricalCrossentropy):

            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def fit(self, x, y, batch_size=32, epochs=1, validation_data=None, steps_per_epoch=100):
        self.metrics.init(y)

        train_steps = 1

        if validation_data is not None:
            validation_steps = 1

            x_val, y_val = validation_data

        if batch_size is not None:
            train_steps = len(x) // batch_size

            if train_steps * batch_size < len(x):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(x_val) // batch_size

                if validation_steps * batch_size < len(x_val):
                    validation_steps += 1

        for epoch in range(1, epochs + 1):
            self.loss_function.new_pass()
            self.metrics.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_x = x
                    batch_y = y
                else:
                    batch_x = x[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                output = self.forward(batch_x, training=True)

                data_loss, regularization_loss = self.loss_function.calculate(output, batch_y, regularization=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.prediction(output)
                accuracy = self.metrics.calculate(predictions, batch_y)

                if not step % steps_per_epoch:
                    print(f'epoch: {epoch}/{epochs} - loss: {loss:.5f} - accuracy: {accuracy:.5f}')

                self.backward(output, batch_y)

                self.optimizer.decay_ajustment()
                for layer in self.trainable_layers:
                    self.optimizer.update_parameters(layer)
                self.optimizer.increment_iteration()

            epoch_data_loss, epoch_regularization_loss = self.loss_function.calculate_accumulated(regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.metrics.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

        if validation_data is not None:
            self.evaluate(*validation_data, batch_size)

    def forward(self, x, training):
        layer = None
        self.input_layer.forward(x, training)

        for layer in self.sequence:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.sequence[-1].gradient_inputs = self.softmax_classifier_output.gradient_inputs

            for layer in reversed(self.sequence[:-1]):
                layer.backward(layer.next.gradient_inputs)

            return

        self.loss_function.backward(output, y)

        for layer in reversed(self.sequence):
            layer.backward(layer.next.gradient_inputs)

    def evaluate(self, x_test, y_test, batch_size=None):

        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(x_test) // batch_size
            if validation_steps * batch_size < len(x_test):
                validation_steps += 1

        self.loss_function.new_pass()
        self.metrics.new_pass()

        for step in range(validation_steps):

            if batch_size is None:
                batch_X = x_test
                batch_y = y_test
            else:
                batch_X = x_test[step * batch_size:(step + 1) * batch_size]
                batch_y = y_test[step * batch_size:(step + 1) * batch_size]

            output = self.forward(batch_X, training=False)

            self.loss_function.calculate(output, batch_y)

            predictions = self.output_layer_activation.prediction(output)
            self.metrics.calculate(predictions, batch_y)

        validation_loss = self.loss_function.calculate_accumulated()
        validation_accuracy = self.metrics.calculate_accumulated()

        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    def predict(self, x, batch_size=None):

        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(x) // batch_size

            if prediction_steps * batch_size < len(x):
                prediction_steps += 1

        output = []

        for step in range(prediction_steps):

            if batch_size is None:
                batch_x = x

            else:
                batch_x = x[step * batch_size:(step + 1) * batch_size]

            batch_output = self.forward(batch_x, training=False)

            output.append(batch_output)

        return np.vstack(output)

    def get_parameters(self):
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        # Return a list
        return parameters

    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    # Saves the model
    def save(self, path):
        model = copy.deepcopy(self)

        model.loss_function.new_pass()
        model.metrics.new_pass()

        model.input_layer.__dict__.pop('output', None)
        model.loss_function.__dict__.pop('gradient_inputs', None)

        for layer in model.sequence:
            for property in ['inputs', 'output', 'gradient_inputs', 'gradient_weights', 'gradient_biases']:
                layer.__dict__.pop(property, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model
