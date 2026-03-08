import os

import tensorflow as tf


def _build_simple_cnn(input_shape):
    """
    Build a very small CNN for binary classification.
    This is NOT a medically useful model – it only exists
    so the web app can function end‑to‑end without the
    original trained weights.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # We don't need to train the model for the UI to work;
    # random initial weights are enough for demonstration.
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def ensure_malaria_model(path: str = "models/malaria.keras") -> None:
    """
    Ensure that a Keras model file for malaria exists.
    If not, create a tiny randomly‑initialized CNN and save it.
    """
    if os.path.exists(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = _build_simple_cnn((36, 36, 3))
    model.save(path)


def ensure_pneumonia_model(path: str = "models/pneumonia.keras") -> None:
    """
    Ensure that a Keras model file for pneumonia exists.
    If not, create a tiny randomly‑initialized CNN and save it.
    """
    if os.path.exists(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = _build_simple_cnn((36, 36, 1))
    model.save(path)

