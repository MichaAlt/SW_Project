
import tensorflow as tf

# Regressionsmodell mit zwei Ausgaben: turn und speed
def create_model(input_shape=(5,)):

     # Eingabeschicht mit 5 Sensordaten
    inputs = tf.keras.Input(shape=input_shape)

    # Versteckte Schichten zur Merkmalsextraktion
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    # Ausgabe für die Lenkung
    # tanh begrenzt den Wert auf [-1, 1]
    turn_output = tf.keras.layers.Dense(
        1,
        activation="tanh",
        name="turn"
    )(x)

    # Ausgabe für die Geschwindigkeit
    # sigmoid begrenzt den Wert auf [0, 1]
    speed_output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="speed"
    )(x)

    # Modell erstellen
    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "turn": turn_output,
            "speed": speed_output
        }
    )

    # Modell kompilieren
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