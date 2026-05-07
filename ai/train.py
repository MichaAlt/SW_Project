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
prediction_cfg = config["prediction"]


# Callback-Klasse definieren
class OverfittingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        if prediction_cfg["prediction"] == "classification":

            train_acc = logs.get('SparseCategoricalAccuracy')
            val_acc = logs.get('val_SparseCategoricalAccuracy')

            if train_acc and val_acc:
                gap = train_acc - val_acc

                # Overfitting, wenn der Unterschied zwischen Trainingaccuracy und Validierungaccuracy größer als 5%
                if gap > 0.1:
                    print("\nTraining gestoppt, Overfitting! Gap: ", gap)
                    self.model.stop_training = True
                    
        elif prediction_cfg["prediction"] == "regression":

            train_loss = logs.get('loss')
            val_loss = logs.get('val_loss')

            if train_loss is None or val_loss is None:
                return

            if val_loss > train_loss * 1.4:
                print("\nTraining gestoppt: Overfitting erkannt")
                self.model.stop_training = True

# Callback-Klasse initialisieren
callbacks = OverfittingCallback()

# Trainingsdaten aus der Konfigurationsdatei laden
if prediction_cfg["prediction"] == "classification":
    data_file = train_cfg["data_file_classification"]
elif prediction_cfg["prediction"] == "regression":
    data_file = train_cfg["data_file_regression"]



train_ds, test_ds, train_ds_features = load_data(
    data_file,
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
    #batch_size=train_cfg["batch_size"] # BatchSize wird jetzt in data_loader definiert
    validation_data = test_ds,
    callbacks=[callbacks]
)

# Modell evaluieren
# print("\nModell evaluieren:")
# loss, acc = model.evaluate(test_ds)
# print("Test Accuracy:", acc, "Test Loss:", loss)

# Modell speichern
if prediction_cfg["prediction"] == "classification":
    save_path = train_cfg["model_save_path_classification"]
elif prediction_cfg["prediction"] == "regression":
    save_path = train_cfg["model_save_path_regression"]

model.save(Path.cwd() / "ai" / train_cfg["model_save_path"])
print(f"Modell gespeichert als {save_path}")
