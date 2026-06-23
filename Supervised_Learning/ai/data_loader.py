from pathlib import Path
import pandas as pd
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys


# Projektwurzel SW_Project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config

config = load_config()
feature_scaling_cfg = config["feature_scaling"]
prediction_cfg = config["prediction"]


def load_data(file_path, n_train=3000, batch_size=30):

    # Trainingsdaten auslesen
    data = pd.read_csv(ROOT_DIR /"Supervised_Learning"/ "ai" / file_path)

    # Prediction per Mehrklassifikation
    if prediction_cfg["prediction"] == "classification":
        # Spaltennamen festlegen
        data.columns = ["r_right", "r_right_front", "r_front","r_left_front", "r_left", "action"]

        # Mapping der Aktionen zu Integer
        mapping = {"W": 0,"A": 1,"D": 2,"W+A": 3,"W+D": 4}

        data["action"] = data["action"].map(mapping)

        # In Features und Labels aufteilen
        x = data.drop("action", axis=1).values
        y = data["action"].values

    # Prediction per Regression
    elif prediction_cfg["prediction"] == "regression":
        # Spaltennamen festlegen
        data.columns = ["r_right", "r_right_front", "r_front","r_left_front", "r_left", "speed", "steering"]

        # In Features und Labels aufteilen
        x = data.drop(["speed", "steering"], axis=1).values
        y = data[["speed", "steering"]].values


    if feature_scaling_cfg["method"] == 1: # Normalisieren
        x = x / 298.0
    elif feature_scaling_cfg["method"] == 2: # Standardisieren
        mean = x.mean()
        std =  x.std()
        x = (x - mean) / std
        np.save("data_file/mean.npy", mean)
        np.save("data_file/std.npy", std)

    # Gesamtdataset bauen
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    # Aufteilen in Trainings- und Validierungsdaten
    train_ds = ds.take(n_train)
    validation_ds = ds.skip(n_train)
    
    # Batch-Size festlegen und Daten mischen
    train_ds = train_ds.batch(batch_size).shuffle(buffer_size=n_train)
    validation_ds = validation_ds.batch(batch_size)

    return train_ds, validation_ds
