
import tensorflow as tf 
import sys
from pathlib import Path

# Projektwurzel SW_Project
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config

config = load_config()
model_config = config["model"]


# Funktion zum erstellen des Modells mit 5 Input-Knoten und 8 Ausgabe-Knoten
def create_model(model_type, train_ds_features, input_shape=(5,), num_classes= 5): 
    
    match model_type:
        case 1:
            # Modell mit normalen Layern
            model = tf.keras.Sequential([ 
                tf.keras.layers.Input(shape=input_shape), # Input Layer mit 5 Features
                tf.keras.layers.Dense(256, activation="relu"), # 1. Hidden Layer
                tf.keras.layers.Dense(128, activation="relu"), # 2. Hidden Layer
                tf.keras.layers.Dense(64, activation="relu"),  # 3. Hidden Layer
                tf.keras.layers.Dense(num_classes , activation="softmax") # Ausgabe des Labels mit der hoechsten Wahrscheinlichkeit
            ])

        case 2:
            # Modell mit BatchNormalization-Layer
            model = tf.keras.Sequential([ 
                tf.keras.layers.Input(shape=input_shape), # Input Layer mit 5 Features
                tf.keras.layers.Dense(128, activation="relu"), # 1. Hidden Layer
                tf.keras.layers.BatchNormalization(), # 1. BatchNormalization Layer
                tf.keras.layers.Dense(64, activation="relu"),  # 2. Hidden Layer
                tf.keras.layers.BatchNormalization(), # 2. BatchNormalization Layer
                tf.keras.layers.Dense(num_classes , activation="softmax") # Ausgabe des Labels mit der hoechsten Wahrscheinlichkeit
            ])

        case 3: # Input mit Normalizer-Layer standardisieren # Funktioniert bei Supervised Learning ueberhaupt nicht, bug?
            normalizer = tf.keras.layers.Normalization(axis=-1) # axis: -1= Features seperat standardisieren, None: Alle Features gemeinsam standardisieren, 
            normalizer.adapt(train_ds_features)
        
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                normalizer,
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(num_classes , activation="softmax") 
            ])
            
            
    # Modell kompilieren
    model.compile(optimizer=model_config["optimizer"], loss='sparse_categorical_crossentropy', metrics=['SparseCategoricalAccuracy']) # SparseCategoricalAccuracy, da die Labels als ganze Zahlen (0-7) vorliegen und nicht als One-Hot-Vektoren kodiert sind
    
    return model



