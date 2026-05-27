import tkinter as tk
from tkinter import ttk
from pathlib import Path
import json
import copy
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "Config" / "config.json"

CLASSIFICATION_DATA_DIR = BASE_DIR / "ai" / "data_file" / "classification_data"
REGRESSION_DATA_DIR = BASE_DIR / "ai" / "data_file" / "regression_data"

CLASSIFICATION_MODELS_DIR = BASE_DIR / "ai" / "models_file" / "classification_models"
REGRESSION_MODELS_DIR = BASE_DIR / "ai" / "models_file" / "regression_models"

MAPS_DIR = BASE_DIR / "race_simulation" / "PNG_File"

def main():
    root = tk.Tk()
    root.title("AI Racing GUI")
    root.geometry("1000x650")
    root.minsize(1000, 500)
    root.configure(background="gray")

    menu_frame = tk.Frame(root, bg="#333333")
    menu_frame.pack(side="top", fill="x")

    content_frame = tk.Frame(root, bg="gray")
    content_frame.pack(side="top", fill="both", expand=True)

    classification_frame = create_model_frame(
        content_frame,
        "Classification",
        "classification",
        CLASSIFICATION_DATA_DIR,
        CLASSIFICATION_MODELS_DIR
    )

    regression_frame = create_model_frame(
        content_frame,
        "Regression",
        "regression",
        REGRESSION_DATA_DIR,
        REGRESSION_MODELS_DIR
    )

    frames = {
        "classification": classification_frame,
        "regression": regression_frame
    }

    def show_frame(frame_name):
        for frame in frames.values():
            frame.pack_forget()

        frames[frame_name].pack(fill="both", expand=True)

    ttk.Button(
        menu_frame,
        text="Classification",
        command=lambda: show_frame("classification")
    ).pack(side="left", padx=5, pady=5)

    ttk.Button(
        menu_frame,
        text="Regression",
        command=lambda: show_frame("regression")
    ).pack(side="left", padx=5, pady=5)

    show_frame("classification")
    root.mainloop()


def create_model_frame(parent, title, model_type, data_folder, model_folder):
    frame = tk.Frame(parent, bg="gray")

    tk.Label(
        frame,
        text=f"{title} Model",
        bg="gray",
        font=("Arial", 20)
    ).pack(pady=15)

    tk.Label(frame, text="Select training data", bg="gray").pack()

    data_box = ttk.Combobox(frame, state="readonly")
    data_box["values"] = get_folder_entries(data_folder)

    if data_box["values"]:
        data_box.current(0)
        update_config_data(model_type, data_box.get())

    data_box.model_type = model_type
    data_box.config_key = "data"
    data_box.bind("<<ComboboxSelected>>", config_changed)
    data_box.pack(padx=10, pady=5, fill="x")

    tk.Label(frame, text="Select model", bg="gray").pack()

    model_box = ttk.Combobox(frame, state="readonly")
    model_box["values"] = get_folder_entries(model_folder)

    if model_box["values"]:
        model_box.current(0)
        update_config_model(model_type, model_box.get())

    model_box.model_type = model_type
    model_box.config_key = "model"
    model_box.bind("<<ComboboxSelected>>", config_changed)
    model_box.pack(padx=10, pady=5, fill="x")

    tk.Label(frame, text="Select map", bg="gray").pack()

    map_box = ttk.Combobox(frame, state="readonly")
    map_box["values"] = get_folder_entries(MAPS_DIR)

    if map_box["values"]:
        map_box.current(0)
        update_config_map(map_box.get())

    map_box.config_key = "map"
    map_box.bind("<<ComboboxSelected>>", config_changed)
    map_box.pack(padx=10, pady=5, fill="x")

    tk.Label(frame, text="New model name", bg="gray").pack()

    model_name_text = tk.Text(frame, height=1)
    model_name_text.pack(padx=10, pady=5, fill="x")

    ttk.Button(
        frame,
        text=f"Create new {model_type} model",
        command=lambda: create_model(
            model_name_text,
            model_type,
            model_box,
            model_folder
        )
    ).pack(
        padx=80,
        pady=6,
        ipady=8,
        fill="x"
    )

    ttk.Button(
        frame,
        text=f"Train {model_type} model",
        command=lambda: train_model(model_type, model_box)
    ).pack(
        padx=80,
        pady=6,
        ipady=8,
        fill="x"
    )

    ttk.Button(
        frame,
        text=f"Run {model_type} model",
        command=lambda: run_model(model_type, model_box, map_box)
    ).pack(
        padx=80,
        pady=6,
        ipady=8,
        fill="x"
    )
    tk.Label(frame, text="Select optimizer", bg="gray").pack()

    optimizer_box = ttk.Combobox(frame, state="readonly")

    optimizer_box["values"] = [
        "adam",
        "sgd"
    ]

    optimizer_box.current(0)

    update_config_optimizer(optimizer_box.get())

    optimizer_box.config_key = "optimizer"
    optimizer_box.bind("<<ComboboxSelected>>", config_changed)

    optimizer_box.pack(
        padx=10,
        pady=5,
        fill="x"
    )
    return frame


def get_folder_entries(folder):
    folder.mkdir(parents=True, exist_ok=True)

    return sorted([
        f.name
        for f in folder.iterdir()
        if not f.name.startswith(".")
    ])


def create_model(text_widget, model_type, model_box, model_folder):
    model_name = text_widget.get("1.0", tk.END).strip()

    if model_name == "":
        model_name = "new_model"

    if not model_name.endswith(".keras"):
        model_name += ".keras"

    model_path = model_folder / model_name
    model_path.touch()

    model_box["values"] = get_folder_entries(model_folder)
    model_box.set(model_name)

    update_prediction_type(model_type)
    update_config_model(model_type, model_name)

    print(f"{model_type} model created: {model_path}")


def train_model(model_type, model_box):
    update_prediction_type(model_type)

    if model_box.get():
        update_config_model(model_type, model_box.get())

    train_path = BASE_DIR / "ai" / "train.py"

    subprocess.run(
        [sys.executable, str(train_path)],
        cwd=str(BASE_DIR)
    )

def update_config_optimizer(optimizer_name):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    config.setdefault("model", {})

    config["model"]["optimizer"] = optimizer_name

    save_config(config, backup)
def run_model(model_type, model_box, map_box):
    update_prediction_type(model_type)

    if model_box.get():
        update_config_model(model_type, model_box.get())

    if map_box.get():
        update_config_map(map_box.get())

    ai_run_path = BASE_DIR / "race_simulation" / "ai_run.py"
    race_simulation_dir = BASE_DIR / "race_simulation"

    if not ai_run_path.exists():
        print(f"ai_run.py not found: {ai_run_path}")
        return

    print(f"Running {model_type} model...")

    subprocess.run(
        [sys.executable, str(ai_run_path)],
        cwd=str(race_simulation_dir)
    )


def config_changed(event):
    widget = event.widget

    if widget.config_key == "data":
        update_config_data(widget.model_type, widget.get())

    elif widget.config_key == "model":
        update_config_model(widget.model_type, widget.get())

    elif widget.config_key == "map":
        update_config_map(widget.get())

    elif widget.config_key == "optimizer":
        update_config_optimizer(widget.get())
def load_config():
    try:
        with open(CONFIG_PATH, "r") as file:
            return json.load(file)
    except Exception as error:
        print("Error loading config:", error)
        return None


def save_config(config, backup):
    try:
        with open(CONFIG_PATH, "w") as file:
            json.dump(config, file, indent=4)
    except Exception as error:
        print("Error saving config:", error)

        with open(CONFIG_PATH, "w") as file:
            json.dump(backup, file, indent=4)


def update_prediction_type(model_type):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    config.setdefault("prediction", {})
    config["prediction"]["prediction"] = model_type

    save_config(config, backup)


def update_config_data(model_type, file_name):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)
    config.setdefault("train", {})

    if model_type == "classification":
        config["train"]["data_file_classification"] = str(
            Path("data_file") / "classification_data" / file_name
        )

    elif model_type == "regression":
        config["train"]["data_file_regression"] = str(
            Path("data_file") / "regression_data" / file_name
        )

    save_config(config, backup)


def update_config_model(model_type, model_name):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    config.setdefault("train", {})
    config.setdefault("ai_run", {})

    if model_type == "classification":
        train_path = Path("models_file") / "classification_models" / model_name
        run_path = Path("..") / "ai" / "models_file" / "classification_models" / model_name

        config["train"]["model_save_path_classification"] = str(train_path)
        config["ai_run"]["model_path_classification"] = str(run_path)

    elif model_type == "regression":
        train_path = Path("models_file") / "regression_models" / model_name
        run_path = Path("..") / "ai" / "models_file" / "regression_models" / model_name

        config["train"]["model_save_path_regression"] = str(train_path)
        config["ai_run"]["model_path_regression"] = str(run_path)

    save_config(config, backup)


def update_config_map(map_name):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    config.setdefault("ai_run", {})

    map_path = str(Path("PNG_File") / map_name)

    config["ai_run"]["map_file"] = map_path
    config["ai_run"]["map_path"] = map_path

    save_config(config, backup)


if __name__ == "__main__":
    main()