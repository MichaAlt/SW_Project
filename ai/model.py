
import tensorflow as tf 
import numpy as np

# Ueberlegung fuer das Modell:
    # 1: 4 Ausgabewerte (W, A, S, D), Ausgabelayer mit Sigmoid-Aktivierungsfunktion da es sich um eine MultiLabel-Ausgabe handelt (mehrere Ausgaben gleichzeitig moeglich) (One-Hot-Codierung?)
    # 2: Mapping der Aktionen zu einem einzigen Wert (z.B. W=0, A=1, S=2, D=3, W+A=4, W+D=5, S+A=6, S+D=7) und Verwendung von Softmax-Aktivierungsfunktion im Ausgabelayer, 
    #    nur eine Aktion pro Schritt moeglich (nur eine Ausgabe moeglich)


# Funktion zum erstellen des Modells mit 5 Input-Knoten und 8 Ausgabe-Knoten
def create_model(input_shape=(5,), num_classes=8): 
    model = tf.keras.Sequential([ 
        tf.keras.layers.Dense(256, activation="relu", input_shape=(5,)), # Erste Schicht mit 256 Neuronen und ReLU-Aktivierungsfunktion
        tf.keras.layers.Dense(128, activation="relu"), # Zweite Schicht mit 128 Neuronen und ReLU-Aktivierungsfunktion
        tf.keras.layers.Dense(num_classes, activation="softmax") # Ausgabe des Labels mit der hoechsten Wahrscheinlichkeit
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model




