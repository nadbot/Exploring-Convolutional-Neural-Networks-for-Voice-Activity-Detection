import numpy as np
import tensorflow.keras.callbacks
from matplotlib import pyplot as plt


def plot_history(history: tensorflow.keras.callbacks.History) -> None:
    """
    Method to plot the training and validation accuracy over the different epochs.
    Args:
        history: history returned by training the model
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def train(model: tensorflow.keras.models, x_train: np.ndarray, y_train: np.ndarray, epochs: int) -> \
        tensorflow.keras.models:
    """
    Train the model by compiling it and then fitting it on the training data.
    Args:
        model: The model to be trained
        x_train: Training data features
        y_train: Training data labels
        epochs: Number of epochs to train on
    """
    model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)

    # plot the accuracy over the epochs
    plot_history(history)
    return model
