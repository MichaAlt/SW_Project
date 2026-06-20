from pathlib import Path
import sys

from SW_Project.xia.ai.data_loader import load_data
from SW_Project.xia.ai.model import create_model

# Projektwurzel SW_Project
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from SW_Project.xia.Config.config_loader import load_config

config = load_config()
train_cfg = config["train"]

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
model.save(train_cfg["model_save_path"])
print(f"Modell gespeichert als {train_cfg['model_save_path']}")