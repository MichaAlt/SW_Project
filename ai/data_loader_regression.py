import pandas as pd
import tensorflow as tf
from pathlib import Path
import sys

# Projektwurzel SW_Project
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config

config = load_config()
feature_scaling_cfg = config["feature_scaling"]


def load_data(file_path, n_train=3000, batch_size=30):

    # Trainingsdaten auslesen
    data = pd.read_csv(file_path)

    # Spaltennamen festlegen
    data.columns = ["r_right", "r_right_front", "r_front","r_left_front", "r_left", "speed", "steering"]

    # In Features und Labels aufteilen
    x = data.drop(["speed", "steering"], axis=1).values
    y = data[["speed", "steering"]].values

    if feature_scaling_cfg["method"] == 1: # Normalisieren
        x = x / 298.0
    elif feature_scaling_cfg["method"] == 2: # Standardisieren
        x = (x - x.mean()) / x.std() 

    # Gesamtdataset bauen
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    # Train/Test Split
    train_ds = ds.take(n_train)
    test_ds = ds.skip(n_train)
    
    train_ds_features = ds.take(n_train).map(lambda x, y: x).batch(batch_size)

    # Batch-Size festlegen und Daten mischen
    train_ds = train_ds.batch(batch_size).shuffle(buffer_size=len(x))
    test_ds = test_ds.batch(batch_size)

    return train_ds, test_ds, train_ds_features
