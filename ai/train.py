from pathlib import Path
import sys

from data_loader import load_data
from model import create_model

# 项目根目录 SW_Project
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Config.config_loader import load_config

config = load_config()
train_cfg = config["train"]

# 如果 config 里没改，就先用组员当前这个数据文件
data_file = train_cfg.get("data_file", "data_file/training_data_map5_2.csv")
n_train = train_cfg.get("n_train", 3000)

x_train, y_train, x_test, y_test = load_data(
    data_file,
    n_train
)

model = create_model()

print("Modell trainieren...")
epochs = train_cfg.get("epochs", 150)
batch_size = train_cfg.get("batch_size", 20)

model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size
)

print("\nModell evaluieren...")
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc, "Test Loss:", loss)

# 先用组员现在的新文件名，避免覆盖旧模型
model_save_path = train_cfg.get("model_save_path", "models_file/model5.keras")
model.save(model_save_path)
print(f"Modell gespeichert als {model_save_path}")