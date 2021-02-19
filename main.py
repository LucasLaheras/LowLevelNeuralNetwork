import nnfs
from nnfs.datasets import *

from activation_function import *
from layers import *
from metrics import *
from model import Sequential
from optimizer import *
import os
import cv2

nnfs.init()

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []


    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(
                        os.path.join(path, dataset, label, file),
                        cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test


# Create dataset
X, y, X_test, y_test = create_data_mnist( 'fashion_mnist_images' )
# Shuffle the training dataset
keys = np.array( range (X.shape[ 0 ]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]
# Scale and reshape samples
X = (X.reshape(X.shape[ 0 ], - 1 ).astype(np.float32) - 127.5 ) / 127.5
X_test = (X_test.reshape(X_test.shape[ 0 ], - 1 ).astype(np.float32) - 127.5 ) / 127.5
# Instantiate the model
model = Sequential()
# Add layers
model.add(Layer_Dense(X.shape[ 1 ], 128 ))
model.add(Activation_ReLU())
model.add(Layer_Dense( 128 , 128 ))
model.add(Activation_ReLU())
model.add(Layer_Dense( 128 , 10 ))
model.add(Activation_softmax())
# Set loss, optimizer and accuracy objects
model.compile(loss = Loss_CategoricalCrossentropy(),optimizer=Optimizer_Adam(decay=1e-4),metrics = Accuracy_Categorical())
# model.fit(X, y, validation_data = (X_test, y_test), epochs = 10 , batch_size = 128 , steps_per_epoch = 100 )
# model.save('fashion_mnist.model')
# model.evaluate(X_test, y_test)
model = model.load('fashion_mnist.model')
model.evaluate(X_test, y_test)
