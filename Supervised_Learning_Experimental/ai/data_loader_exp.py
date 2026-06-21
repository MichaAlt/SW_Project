import pandas as pd
import numpy as np

def load_data(file_path, n_train=3000):
    data = pd.read_csv(file_path, header=None)

    data.columns = [
        "r_right",
        "r_right_front",
        "r_front",
        "r_left_front",
        "r_left",
        "pos_x",
        "pos_y",
        "angle",
        "turn_angle",
        "speed_norm" 
    ]

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    data_train = data.iloc[:n_train]
    data_test = data.iloc[n_train:]

    # Eingabe: nur 5 Sensorwerte verwenden
    x_train = data_train[
        ["r_right", "r_right_front", "r_front", "r_left_front", "r_left"]
    ].values.astype("float32")

    x_test = data_test[
        ["r_right", "r_right_front", "r_front", "r_left_front", "r_left"]
    ].values.astype("float32")

    # Ausgabe 1: Lenkwinkel
    y_train_turn = data_train[["turn_angle"]].values.astype("float32")
    y_test_turn = data_test[["turn_angle"]].values.astype("float32")

    # Ausgabe 2: Geschwindigkeit
    y_train_speed = data_train[["speed_norm"]].values.astype("float32")
    y_test_speed = data_test[["speed_norm"]].values.astype("float32")

    # 5 Sensorwerte normalisieren
    x_train[:, 0:5] = x_train[:, 0:5] / 298.0
    x_test[:, 0:5] = x_test[:, 0:5] / 298.0

    # turn_angle auf den Bereich -1 bis 1 normalisieren
    y_train_turn = np.clip(y_train_turn / 180.0, -1.0, 1.0)
    y_test_turn = np.clip(y_test_turn / 180.0, -1.0, 1.0)

    # speed_norm sicherheitshalber auf den Bereich 0 bis 1 begrenzen
    y_train_speed = np.clip(y_train_speed, 0.0, 1.0)
    y_test_speed = np.clip(y_test_speed, 0.0, 1.0)

    return (
        x_train,
        {"turn": y_train_turn, "speed": y_train_speed},
        x_test,
        {"turn": y_test_turn, "speed": y_test_speed}
    )