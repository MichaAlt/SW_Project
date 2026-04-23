from pathlib import Path
import sys

from data_loader import load_data
from model import create_model

# Projektwurzel SW_Project
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config

config = load_config()
train_cfg = config["train"]

# Trainingsdaten aus der Konfigurationsdatei laden
train_ds, test_ds = load_data(
    train_cfg["data_file"],
    train_cfg["n_train"],
    train_cfg["batch_size"]
)

# Modell erstellen
model = create_model()

# Modell trainieren
print("Modell trainieren...")
model.fit(
    train_ds,
    epochs=train_cfg["epochs"],
    #batch_size=train_cfg["batch_size"]
)

# Modell evaluieren
print("\nModell evaluieren:")
loss, acc = model.evaluate(test_ds)
print("Test Accuracy:", acc, "Test Loss:", loss)

# Modell speichern
model.save(train_cfg["model_save_path"])
print(f"Modell gespeichert als {train_cfg['model_save_path']}")