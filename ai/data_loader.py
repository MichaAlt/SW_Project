import pandas as pd # Pandas dient zur Datenverarbeitung und -analyse, unter anderem zum Einlesen von CSV-Dateien

def load_data(file_path):
    # Trainingsdaten auslesen
    data = pd.read_csv(file_path) 

    # Spaltennamen der Trainingsdaten festlegen
    data.columns = ["r_right", "r_right_front", "r_front", "r_left_front", "r_left", "action"] 

    # mapping = { "W": 0, "A": 1, "S": 2, "D": 3, "W+A": 4, "W+D": 5,"S+A": 6, "S+D": 7 } # Mapping der Aktionen zu numerischen Werten fuer TensorFlow
    mapping = { "W": 0, "A": 1, "D": 2, "W+A": 3, "W+D": 4 }
    
    # Aktionen in numerische Werte umwandeln
    data["action"] = data["action"].map(mapping) 

    #data = data.sample(frac=1).reset_index(drop=True) # Trainingsdaten mischen, (frac=1) = 100% der Daten mischen, reset_index(drop=True) = Datenindex nach dem Mischen zurücksetzen
    n_data = 3000
    data_train = data.iloc[:n_data] #data_train = data.take(400) # Erste 400 Zeilen der Trainingsdaten fuer das Training verwenden
    data_test = data.iloc[n_data:] #ds_test = data.skip(400) # Restliche Zeilen der Trainingsdaten fuer das Testen verwenden
    print(data.head()) # Erste 6 Zeilen der Trainingsdaten nach dem Mischen printen

    # Aufteilung der Trainingsdaten in Features und Labels
    x_train = data_train.drop("action", axis = 1).values # axis = 1, um Spalte mit den Namen "action" zu loeschen bzw. zu "droppen"
    y_train = data_train["action"].values
    x_test =  data_test.drop("action", axis = 1).values
    y_test = data_test["action"].values

    # Daten normalisieren
    x_train = x_train / 298.0 
    x_test = x_test / 298.0 
    
    return x_train, y_train, x_test, y_test