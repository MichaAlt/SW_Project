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


class OverfittingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        if prediction_cfg["prediction"] == "classification":
            train_acc = logs.get("SparseCategoricalAccuracy")
            val_acc = logs.get("val_SparseCategoricalAccuracy")

            if train_acc is not None and val_acc is not None:
                gap = train_acc - val_acc

                if gap > 0.1:
                    print("\nTraining gestoppt, Overfitting! Gap:", gap)
                    self.model.stop_training = True

        elif prediction_cfg["prediction"] == "regression":
            train_loss = logs.get("loss")
            val_loss = logs.get("val_loss")

            if train_loss is None or val_loss is None:
                return

            if val_loss > train_loss * 1.4:
                print("\nTraining gestoppt: Overfitting erkannt")
                self.model.stop_training = True


def get_optimizer():
    optimizer_name = model_cfg.get("optimizer", "adam").lower()

    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam()

    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD()

    raise ValueError(f"Unbekannter Optimizer: {optimizer_name}")


callbacks = OverfittingCallback()

if prediction_cfg["prediction"] == "classification":
    data_file = train_cfg["data_file_classification"]

elif prediction_cfg["prediction"] == "regression":
    data_file = train_cfg["data_file_regression"]

else:
    raise ValueError("prediction muss classification oder regression sein")


train_ds, test_ds, train_ds_features = load_data(
    data_file,
    train_cfg["n_train"],
    train_cfg["batch_size"]
)

model = create_model(
    model_type=train_cfg["model_type"],
    train_ds_features=train_ds_features
)

optimizer = get_optimizer()

if prediction_cfg["prediction"] == "classification":
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="SparseCategoricalAccuracy")]

elif prediction_cfg["prediction"] == "regression":
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanAbsoluteError(name="mae")]

else:
    raise ValueError("prediction muss classification oder regression sein")

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)

print(f"Optimizer: {model_cfg.get('optimizer', 'adam')}")
print("Modell trainieren...")

history = model.fit(
    train_ds,
    epochs=train_cfg["epochs"],
    validation_data=test_ds,
    callbacks=[callbacks]
)

if prediction_cfg["prediction"] == "classification":
    save_path = Path.cwd() / "ai" / train_cfg["model_save_path_classification"]

elif prediction_cfg["prediction"] == "regression":
    save_path = Path.cwd() / "ai" / train_cfg["model_save_path_regression"]

else:
    raise ValueError("prediction muss classification oder regression sein")

model.save(save_path)
print(f"Modell gespeichert als {save_path}")