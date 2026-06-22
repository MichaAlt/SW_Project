from pathlib import Path
import sys

from data_loader_exp import load_data
from model_exp import create_model

# Projektwurzel SW_Project
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config

config = load_config()
train_cfg = config["train_regression_exp"]

# Trainingsdaten aus der Konfigurationsdatei laden
x_train, y_train, x_test, y_test = load_data(
    train_cfg["data_file"],
    train_cfg["n_train"],
)

# Modell erstellen
model = create_model()

# Modell trainieren
print("Modell trainieren...")
history = model.fit(
    x_train,
    y_train,
    epochs=train_cfg["epochs"],
    batch_size=train_cfg["batch_size"]
)

# Modell evaluieren
print("\nModell evaluieren:")
results = model.evaluate(x_test, y_test, return_dict=True)

for key, value in results.items():
    print(f"{key}: {value}")

# Modell speichern
save_path = Path.cwd() / ".." / "Supervised_Learning_Experimental" / "ai" / train_cfg["model_save_path"]
model.save(save_path)
print("Modell gespeichert!") 