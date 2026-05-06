
import tensorflow as tf 
import numpy as np

# Ueberlegung fuer das Modell:
    # 1: 4 Ausgabewerte (W, A, S, D), Ausgabelayer mit Sigmoid-Aktivierungsfunktion da es sich um eine MultiLabel-Ausgabe handelt (mehrere Ausgaben gleichzeitig moeglich) (One-Hot-Codierung?)
    # 2: Mapping der Aktionen zu einem einzigen Wert (z.B. W=0, A=1, S=2, D=3, W+A=4, W+D=5, S+A=6, S+D=7) und Verwendung von Softmax-Aktivierungsfunktion im Ausgabelayer, 
    #    nur eine Aktion pro Schritt moeglich (nur eine Ausgabe moeglich)

"""
# Funktion zum erstellen des Modells mit 5 Input-Knoten und 8 Ausgabe-Knoten
def create_model(input_shape=(5,), num_classes= 5): 
    
    # Modell mit BatchNormalization-Layer
    model = tf.keras.Sequential([ 
        tf.keras.layers.Input(shape=input_shape), # Input Layer mit 5 Features
        tf.keras.layers.Dense(128, activation="relu"), # 1. Hidden Layer
        tf.keras.layers.BatchNormalization(), # 1. BatchNormalization Layer
        tf.keras.layers.Dense(64, activation="relu"),  # 2. Hidden Layer
        tf.keras.layers.BatchNormalization(), # 2. BatchNormalization Layer
        tf.keras.layers.Dense(num_classes , activation="softmax") # Ausgabe des Labels mit der hoechsten Wahrscheinlichkeit
    ])

    
    # Modell mit normalen Layern
    model = tf.keras.Sequential([ 
        tf.keras.layers.Input(shape=input_shape), # Input Layer mit 5 Features
        tf.keras.layers.Dense(256, activation="relu"), # 1. Hidden Layer
        tf.keras.layers.Dense(128, activation="relu"), # 2. Hidden Layer
        tf.keras.layers.Dense(64, activation="relu"),  # 3. Hidden Layer
        tf.keras.layers.Dense(num_classes , activation="softmax") # Ausgabe des Labels mit der hoechsten Wahrscheinlichkeit
    ])
    
    # Modell kompilieren
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['SparseCategoricalAccuracy']) # SparseCategoricalAccuracy, da die Labels als ganze Zahlen (0-7) vorliegen und nicht als One-Hot-Vektoren kodiert sind
    
    return model
 """
def create_model(input_shape=(6,)):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Dense(256, activation="relu"),

        tf.keras.layers.Dense(128, activation="relu"),

        tf.keras.layers.Dense(64, activation="relu"),

        tf.keras.layers.Dense(2)

    ])

    model.compile(

        optimizer="sgd",

        loss="mse",

        metrics=["mae"]

    )

    return model


