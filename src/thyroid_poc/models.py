from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


def alexnet(input_shape: tuple[int, int, int] = (224, 224, 3)) -> Model:
    return tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(96, 11, strides=4, activation="relu"),
            layers.MaxPooling2D(3, strides=2),
            layers.BatchNormalization(),
            layers.Conv2D(256, 5, padding="same", activation="relu"),
            layers.MaxPooling2D(3, strides=2),
            layers.BatchNormalization(),
            layers.Conv2D(384, 3, padding="same", activation="relu"),
            layers.Conv2D(384, 3, padding="same", activation="relu"),
            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(3, strides=2),
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="alexnet_binary",
    )


def transfer_model(
    architecture: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    *,
    pretrained: bool = True,
) -> Model:
    builders = {
        "inception_v3": tf.keras.applications.InceptionV3,
        "inception_resnet_v2": tf.keras.applications.InceptionResNetV2,
    }
    if architecture not in builders:
        raise ValueError(f"Unsupported architecture: {architecture}")
    base = builders[architecture](
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=input_shape,
    )
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.inception_v3.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs, name=f"{architecture}_binary")


def build_model(name: str, input_shape: tuple[int, int, int] = (224, 224, 3)) -> Model:
    if name == "alexnet":
        return alexnet(input_shape)
    return transfer_model(name, input_shape)

