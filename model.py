from tensorflow.keras import layers, models


def create_rgb_model() -> models.Sequential:
    """
    Create a simple tensorflow model for learning rgb spectrogram input.
    This model is as described in the original paper.
    """
    shape = (128, 32, 1)

    model = models.Sequential(name="LeNet5")
    model.add(layers.Conv2D(20, 5, activation='tanh', input_shape=shape, padding='same'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(50, 5, activation='tanh', padding='same'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(500))
    model.add(layers.ReLU())
    model.add(layers.Dense(2))
    model.add(layers.Softmax())

    model.summary()
    return model
