import numpy as np


def book_chapter_5(softmax_outputs, class_targets):
    # Calculate values along second axis (axis of index 1)
    predictions = np.argmax(softmax_outputs, axis=1)
    # If targets are one-hot encoded - convert them
    if len(class_targets.shape) == 2:
        class_targets = np.argmax(class_targets, axis=1)
    # True evaluates to 1; False to 0
    accuracy = np.mean(predictions == class_targets)

    return accuracy