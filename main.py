import numpy as np
import layers
import activation_function
from model import Sequential
import nnfs
from nnfs import datasets
from optimizer import *
from activation_function import *

nnfs.init()

X, y = datasets.spiral_data(samples=100, classes=3)

print(np.argmax([[1, 2, 3, 4, 5], [1, 7, 3, 4, 5]], axis=1))

model = Sequential([
    layers.Layer_Dense(2, 64),
    activation_function.Activation_ReLU(),
    layers.Layer_Dense(64, 3)
])

model.compile(optimizer=Optimizer_stochastic_gradient_descent(decay=1e-4, momentum=0.95), loss=Activation_Softmax_Loss_CategoricalCrossentropy())

model.fit(X, y, 10000)
