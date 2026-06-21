
import tensorflow as tf

def create_model(input_shape=(5,)):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    turn_output = tf.keras.layers.Dense(
        1,
        activation="tanh",
        name="turn"
    )(x)

    speed_output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="speed"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "turn": turn_output,
            "speed": speed_output
        }
    )

    model.compile(
        optimizer="adam",
        loss={
            "turn": "mse",
            "speed": "mse"
        },
        metrics={
            "turn": ["mae"],
            "speed": ["mae"]
        }
    )

    return model