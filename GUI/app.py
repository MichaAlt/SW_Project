import tkinter as tk
from tkinter import ttk
from pathlib import Path
import json
import copy
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "Config" / "config.json"

CLASSIFICATION_DATA_DIR = BASE_DIR / "Supervised_Learning" / "ai" / "data_file" / "classification_data"
REGRESSION_DATA_DIR = BASE_DIR / "Supervised_Learning" / "ai" / "data_file" / "regression_data"
CLASSIFICATION_MODELS_DIR = BASE_DIR / "Supervised_Learning" / "ai" / "models_file" / "classification_models"
REGRESSION_MODELS_DIR = BASE_DIR / "Supervised_Learning" / "ai" / "models_file" / "regression_models"

PPO_MODELS_DIR = BASE_DIR / "Reinforcement_Learning" / "models_file"/ "PPO_models_file"
DDPG_MODELS_DIR = BASE_DIR / "Reinforcement_Learning" / "models_file"/ "DDPG_models_file"
SAC_MODELS_DIR = BASE_DIR / "Reinforcement_Learning" / "models_file"/ "SAC_models_file"

MAPS_DIR = BASE_DIR / "PNG_File"

def main():
    root = tk.Tk()
    root.title("GUI")
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

    rl_frame = create_rl_frame(content_frame)

    frames = {
        "classification": classification_frame,
        "regression": regression_frame,
        "reinforcement_learning" : rl_frame
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

    ttk.Button(
        menu_frame,
        text="Reinforcement_Learning",
        command=lambda: show_frame("reinforcement_learning")
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
    if model_name == "":model_name = "new_model"
    if not model_name.endswith(".keras"):model_name += ".keras"
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

    train_path = BASE_DIR / "Supervised_Learning" / "ai" / "train.py"

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

    ai_run_path = BASE_DIR / "Supervised_Learning" / "race_simulation" / "ai_run.py"
    race_simulation_dir = BASE_DIR  / "Supervised_Learning" / "race_simulation"

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

    map_path = str(Path("../../PNG_File") / map_name)

    config["ai_run"]["map_file"] = map_path
    config["ai_run"]["map_path"] = map_path

    save_config(config, backup)







def create_rl_frame(parent):

    frame = tk.Frame(parent, bg="gray")
    tk.Label(frame,text="Reinforcement Learning",bg="gray",font=("Arial", 20)).pack(pady=15)
    tk.Label(frame,text="Select RL Algorithm",bg="gray").pack()
    algorithm_box = ttk.Combobox(frame,state="readonly")
    algorithm_box["values"] = ["PPO","DDPG","SAC"]
    algorithm_box.current(0)

    update_config_algorithm_rl(algorithm_box.get())

    algorithm_box.pack(padx=10,pady=5,fill="x")
    tk.Label(frame,text="Select training model",bg="gray").pack()
    train_model_box = ttk.Combobox(frame,state="readonly")
    train_model_box["values"] = get_model_entries("PPO")

    if train_model_box["values"]:
        train_model_box.current(0)
        update_config_train_model_rl("PPO",train_model_box.get())

    train_model_box.pack(padx=10,pady=5,fill="x")
    tk.Label(frame,text="Select run model",bg="gray").pack()
    run_model_box = ttk.Combobox(frame,state="readonly")
    run_model_box["values"] = get_model_entries("PPO")

    if run_model_box["values"]:
        run_model_box.current(0)
        update_config_run_model_rl("PPO",run_model_box.get())

    run_model_box.pack(padx=10,pady=5,fill="x")
    tk.Label(frame,text="Select map",bg="gray").pack()
    map_box = ttk.Combobox(frame,state="readonly")
    map_box["values"] = get_folder_entries(MAPS_DIR)

    if map_box["values"]:
        map_box.current(0)
        update_config_map_rl(map_box.get())

    map_box.pack(padx=10,pady=5,fill="x")
    tk.Label(frame,text="New model name",bg="gray").pack()
    model_name_text = tk.Text(frame,height=1)
    model_name_text.pack(padx=10,pady=5,fill="x")
    ttk.Button(frame,text="Create new RL model",command=lambda: create_model_rl(model_name_text,algorithm_box,train_model_box,run_model_box)).pack(padx=80,pady=6,ipady=8,fill="x")
    ttk.Button(frame,text="Train RL model",command=train_model_rl).pack(padx=80,pady=6,ipady=8,fill="x")
    ttk.Button(frame,text="Run RL model",command=run_model_rl).pack(padx=80,pady=6,ipady=8,fill="x")

    def algorithm_changed(event):

        algorithm = algorithm_box.get()
        values = get_model_entries(algorithm)
        train_model_box["values"] = values
        run_model_box["values"] = values

        if values:
            train_model_box.current(0)
            run_model_box.current(0)
            update_config_train_model_rl(algorithm,train_model_box.get())
            update_config_run_model_rl(algorithm,run_model_box.get())

        update_config_algorithm_rl(algorithm)

    algorithm_box.bind("<<ComboboxSelected>>",algorithm_changed)

    train_model_box.bind(
        "<<ComboboxSelected>>",
        lambda event:
        update_config_train_model_rl(
            algorithm_box.get(),
            train_model_box.get()
        )
    )

    run_model_box.bind("<<ComboboxSelected>>",lambda event:update_config_run_model_rl(algorithm_box.get(),run_model_box.get()))
    map_box.bind("<<ComboboxSelected>>",lambda event:update_config_map_rl(map_box.get()))

    return frame


def get_model_dir(algorithm):

    match algorithm:
        case "PPO":
            return PPO_MODELS_DIR
        case "DDPG":
            return DDPG_MODELS_DIR
        case "SAC":
            return SAC_MODELS_DIR


def get_model_entries(algorithm):

    folder = get_model_dir(algorithm)
    folder.mkdir(parents=True,exist_ok=True)

    return sorted([
        f.name
        for f in folder.iterdir()
        if not f.name.startswith(".")
        ])


def create_model_rl(text_widget,algorithm_box,train_model_box,run_model_box):

    model_name = text_widget.get("1.0",tk.END).strip()
    if model_name == "":model_name = "new_model"
    if not model_name.endswith(".zip"):model_name += ".zip"
    algorithm = algorithm_box.get()
    model_dir = get_model_dir(algorithm)
    model_path = model_dir / model_name
    model_path.touch()
    values = get_model_entries(algorithm)
    train_model_box["values"] = values
    run_model_box["values"] = values
    train_model_box.set(model_name)
    run_model_box.set(model_name)

    update_config_train_model_rl(algorithm,model_name)
    update_config_run_model_rl(algorithm,model_name)
    print(f"{algorithm} model created: {model_path}")

def train_model_rl():

    train_path = (BASE_DIR / "Reinforcement_Learning" / "train_rl.py")
    subprocess.run([sys.executable, str(train_path)],cwd=str(BASE_DIR/"Reinforcement_Learning"))


def run_model_rl():

    run_path = (BASE_DIR  / "Reinforcement_Learning"/ "run_rl.py") 
    subprocess.run([sys.executable, str(run_path)],cwd=str(BASE_DIR /"Reinforcement_Learning"))

def update_config_algorithm_rl(algorithm):

    config = load_config()

    if config is None:return

    backup = copy.deepcopy(config)

    config["train_rl"]["rl_algorithm"] = algorithm
    config["run_rl"]["rl_algorithm"] = algorithm

    save_config(config,backup)


def update_config_train_model_rl(algorithm,model_name):

    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)

    match algorithm:
        case "PPO":
            config["train_rl"]["model_save_path_PPO"] = (f"models_file/PPO_models_file/{model_name}")
        case "DDPG":
            config["train_rl"]["model_save_path_DDPG"] = (f"models_file/DDPG_models_file/{model_name}")
        case "SAC":
            config["train_rl"]["model_save_path_SAC"] = (f"models_file/SAC_models_file/{model_name}")

    save_config(config,backup)


def update_config_run_model_rl(algorithm,model_name):

    config = load_config()
    
    if config is None:
        return

    backup = copy.deepcopy(config)

    match algorithm:
        case "PPO":
            config["run_rl"]["model_load_path_PPO"] = (f"models_file/PPO_models_file/{model_name}")
        case "DDPG":
            config["run_rl"]["model_load_path_DDPG"] = (f"models_file/DDPG_models_file/{model_name}")
        case "SAC":
            config["run_rl"]["model_load_path_SAC"] = (f"models_file/SAC_models_file/{model_name}")

    save_config(config,backup)


def update_config_map_rl(map_name):
    config = load_config()

    if config is None:
        return

    backup = copy.deepcopy(config)
    map_path = str(Path("../PNG_File") / map_name)
    config["train_rl"]["map_file"] = map_path
    config["run_rl"]["map_file"] = map_path
    save_config(config,backup)



if __name__ == "__main__":
    main()