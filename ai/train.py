import tensorflow as tf
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


# Callback-Klasse definieren
class OverfittingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('SparseCategoricalAccuracy')
        val_acc = logs.get('val_SparseCategoricalAccuracy')

        if train_acc and val_acc:
            gap = train_acc - val_acc

            # Overfitting, wenn der Gap zwischen Training- und Validierung - Accuracy größer als 5%
            if gap > 0.05:
                print("\nTraining gestoppt, Overfitting! Gap: ", gap)
                self.model.stop_training = True

# Callback-Klasse initialisieren
callbacks = OverfittingCallback()

# Trainingsdaten aus der Konfigurationsdatei laden
train_ds, test_ds, train_ds_features = load_data(
    train_cfg["data_file"],
    train_cfg["n_train"],
    train_cfg["batch_size"]
)

# Modell erstellen
model = create_model(model_type=train_cfg["model_type"], train_ds_features = train_ds_features)

# Modell trainieren
print("Modell trainieren...")
history = model.fit(
    train_ds,
    epochs=train_cfg["epochs"],
    #batch_size=train_cfg["batch_size"]
    validation_data = test_ds,
    callbacks=[callbacks]
)

# Modell evaluieren
# print("\nModell evaluieren:")
# loss, acc = model.evaluate(test_ds)
# print("Test Accuracy:", acc, "Test Loss:", loss)

# Modell speichern
model.save(train_cfg["model_save_path"])
print(f"Modell gespeichert als {train_cfg['model_save_path']}")