import pandas as pd  # Pandas dient zur Datenverarbeitung und -analyse, unter anderem zum Einlesen von CSV-Dateien

def load_data(file_path, n_train=3000):
    # Trainingsdaten auslesen
    data = pd.read_csv(file_path)

    # Spaltennamen festlegen
    data.columns = ["r_right", "r_right_front", "r_front", "r_left_front", "r_left", "action"]

    # Eingabespalten in numerische Werte umwandeln
    feature_cols = ["r_right", "r_right_front", "r_front", "r_left_front", "r_left"]
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Mapping der Aktionen zu numerischen Werten fuer TensorFlow
    # wichtig: dieses Mapping muss mit ai_run.py identisch sein
    mapping = {
        "W": 0,
        "A": 1,
        "D": 2,
        "W+A": 3,
        "W+D": 4
    }

    # Aktionen in numerische Werte umwandeln
    data["action"] = data["action"].map(mapping)

    # Zeilen mit ungueltigen Werten entfernen
    data = data.dropna()

    # Trainingsdaten mischen
    data = data.sample(frac=1).reset_index(drop=True)

    # Aufteilung in Trainings- und Testdaten
    data_train = data.iloc[:n_train]
    data_test = data.iloc[n_train:]

    # Aufteilung der Trainingsdaten in Features und Labels
    x_train = data_train.drop("action", axis=1).values.astype("float32")
    y_train = data_train["action"].values.astype("int32")
    x_test = data_test.drop("action", axis=1).values.astype("float32")
    y_test = data_test["action"].values.astype("int32")

    # Daten normalisieren
    x_train = x_train / 298.0
    x_test = x_test / 298.0

    return x_train, y_train, x_test, y_test