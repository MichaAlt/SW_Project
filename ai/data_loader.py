"""
import pandas as pd  # Pandas dient zur Datenverarbeitung und -analyse, unter anderem zum Einlesen von CSV-Dateien

def load_data(file_path, n_train=3000):
    # Trainingsdaten auslesen
    data = pd.read_csv(file_path)

    # Spaltennamen festlegen
    data.columns = ["r_right", "r_right_front", "r_front", "r_left_front", "r_left", "action"]
   
    # Mapping der Aktionen zu Integer
    mapping = {"W": 0,"A": 1,"D": 2,"W+A": 3,"W+D": 4}

    data["action"] = data["action"].map(mapping)
    
    # Aufteilung in Trainings- und Testdaten
    data_train = data.iloc[:n_train]
    data_test = data.iloc[n_train:]

    # Aufteilung der Trainingsdaten in Features und Labels
    x_train = data_train.drop("action", axis=1).values
    y_train = data_train["action"].values
    x_test = data_test.drop("action", axis=1).values
    y_test = data_test["action"].values

    # Daten normalisieren
    x_train = x_train / 298.0
    x_test = x_test / 298.0

    return x_train, y_train, x_test, y_test
"""

import pandas as pd
import tensorflow as tf

def load_data(file_path, n_train=3000, batch_size=30):

    # Trainingsdaten auslesen
    data = pd.read_csv(file_path)

    # Spaltennamen festlegen
    data.columns = ["r_right", "r_right_front", "r_front","r_left_front", "r_left", "action"]

    # Mapping der Aktionen zu Integer
    mapping = {"W": 0,"A": 1,"D": 2,"W+A": 3,"W+D": 4}

    data["action"] = data["action"].map(mapping)

    # In Features und Labels aufteilen
    x = data.drop("action", axis=1).values
    y = data["action"].values

    # Normalisierung
    x = x / 298.0
    # Standardisieren
    #x = (x - x.mean()) / x.std()

    # Gesamtdataset bauen
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    # Train/Test Split
    train_ds = ds.take(n_train)
    test_ds = ds.skip(n_train)
    
    train_ds_features = ds.take(n_train).map(lambda x, y: x).batch(batch_size)

    # Batch-Size festlegen und Daten mischen
    train_ds = train_ds.batch(batch_size).shuffle(buffer_size=len(train_ds))
    test_ds = test_ds.batch(batch_size).shuffle(buffer_size=len(test_ds))

    return train_ds, test_ds, train_ds_features

