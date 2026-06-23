import tensorflow as tf
from pathlib import Path
import sys

from data_loader import load_data
from model import create_model

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config

config = load_config()
train_cfg = config["train"]
prediction_cfg = config["prediction"]
model_cfg = config["model"]


# Callback zum erkennen von Overfitting und zum fruezeitigen beenden des Trainings
class OverfittingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        # Callback fuer Mehrklassifikation
        if prediction_cfg["prediction"] == "classification":
            train_acc = logs.get("SparseCategoricalAccuracy")
            val_acc = logs.get("val_SparseCategoricalAccuracy")

            if train_acc is not None and val_acc is not None:
                gap = train_acc - val_acc

                # Wenn die Luecke zwischen Traings- und Validierungs-Accuracy groesser als 0.1 ist,
                # wird Training gestoptt
                if gap > 0.1:
                    print("\nTraining gestoppt, Overfitting! Gap:", gap)
                    self.model.stop_training = True

        # Callback fuer Regression
        elif prediction_cfg["prediction"] == "regression":
            train_loss = logs.get("loss")
            val_loss = logs.get("val_loss")

            # Ueberspringen wenn train_loss oder val_loss nicht vorhanden
            if train_loss is None or val_loss is None:
                return

            # Wenn Validation-Loss mehr als 40% groesser ist, wird Training gestoppt
            if val_loss > train_loss * 1.4:
                print("\nTraining gestoppt: Overfitting erkannt")
                self.model.stop_training = True


callbacks = OverfittingCallback()

# Datenauswahl nach Prediction-Methode
if prediction_cfg["prediction"] == "classification":
    data_file = train_cfg["data_file_classification"]

elif prediction_cfg["prediction"] == "regression":
    data_file = train_cfg["data_file_regression"]

else:
    raise ValueError("prediction muss classification oder regression sein")

# Trainings- und Validierungsdatasets laden
train_ds, validation_ds= load_data(data_file,train_cfg["n_train"],train_cfg["batch_size"])

# Modell erstellen
model = create_model(model_type=train_cfg["model_type"])

# Modell trainieren
print("Modell trainieren...")
history = model.fit(
    train_ds,
    epochs=train_cfg["epochs"],
    validation_data=validation_ds,
    callbacks=[callbacks]
)


# Speicherpfadauswahl nach Prediction-Methode
if prediction_cfg["prediction"] == "classification":
    save_path = Path.cwd() / ".." / "Supervised_Learning" / "ai" / train_cfg["model_save_path_classification"]

elif prediction_cfg["prediction"] == "regression":
    save_path = Path.cwd() / ".." /"Supervised_Learning" /"ai" / train_cfg["model_save_path_regression"]

else:
    raise ValueError("prediction muss classification oder regression sein")

# Modell speichern
model.save(save_path)
print("Modell gespeichert!")