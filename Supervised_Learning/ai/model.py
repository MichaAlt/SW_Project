import tensorflow as tf 
import sys
from pathlib import Path

# Projektwurzel SW_Project
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config

# Konfigurationen laden
config = load_config()
model_config = config["model"]
prediction_cfg = config["prediction"]


# Output-Aktivierungsfunktion, loss, metrics und Anzahl der Outputs ueber prediction_config festlegen
if prediction_cfg["prediction"] == "classification":
    activation = "softmax"
    loss = 'sparse_categorical_crossentropy'
    metrics = ['SparseCategoricalAccuracy']
    num_outputs = 5

elif prediction_cfg["prediction"] == "regression":
    activation = "linear"
    loss = 'mse'
    metrics = ['mse']
    num_outputs = 2



# Funktion zum erstellen des Modells mit 5 Input-Knoten und 8 Ausgabe-Knoten
def create_model(model_type, input_shape=(5,)): 
    
    # Auswahl des Modells ueber model_type
    match model_type:
        case 1:
            # Modell mit normalen Layern
            model = tf.keras.Sequential([ 
                tf.keras.layers.Input(shape=input_shape), # Input Layer mit 5 Features
                tf.keras.layers.Dense(256, activation="relu"), # 1. Hidden Layer
                tf.keras.layers.Dense(128, activation="relu"), # 2. Hidden Layer
                tf.keras.layers.Dense(64, activation="relu"),  # 3. Hidden Layer
                tf.keras.layers.Dense(num_outputs , activation=activation) # Ausgabelayer
            ])

        case 2:
            # Modell mit BatchNormalization-Layer
            model = tf.keras.Sequential([ 
                tf.keras.layers.Input(shape=input_shape), # Input Layer mit 5 Features
                tf.keras.layers.Dense(128, activation="relu"), # 1. Hidden Layer
                tf.keras.layers.BatchNormalization(), # 1. BatchNormalization Layer
                tf.keras.layers.Dense(64, activation="relu"),  # 2. Hidden Layer
                tf.keras.layers.BatchNormalization(), # 2. BatchNormalization Layer
                tf.keras.layers.Dense(num_outputs , activation=activation) # Ausgabelayer
            ])

        case 3:
            # Modell mit Dropout Layern, um Overfitting vorzubeugen
            model = tf.keras.Sequential([ 
                tf.keras.layers.Input(shape=input_shape), # Input Layer mit 5 Features
                tf.keras.layers.Dropout(0.1), # Dropoutlayer
                tf.keras.layers.Dense(256, activation="relu"), # 1. Hidden Layer
                tf.keras.layers.Dense(128, activation="relu"), # 2. Hidden Layer
                tf.keras.layers.Dense(64, activation="relu"),  # 3. Hidden Layer
                tf.keras.layers.Dense(num_outputs , activation=activation)  # Ausgabelayer
            ])
            
        case 4:
            # Modell mit L2-Regularisierung Layern, um Overfitting vorzubeugen
            model = tf.keras.Sequential([ 
                tf.keras.layers.Input(shape=input_shape), # Input Layer mit 5 Features
                tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)), # 1. Hidden Layer mit L2-Regularisierungslayer
                tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)), # 2. Hidden Layer mit L2-Regularisierungslayer
                tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),  # 3. Hidden Layer mit L2-Regularisierungslayer
                tf.keras.layers.Dense(num_outputs , activation=activation) # Ausgabelayer
            ])
        
            
    # Modell kompilieren
    model.compile(optimizer=model_config["optimizer"], loss=loss, metrics=metrics) 
    
    return model



